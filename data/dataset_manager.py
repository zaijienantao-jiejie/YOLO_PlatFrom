# -*- coding: utf-8 -*-
"""数据集管理：创建、导入、导出、划分、版本快照、AI 标注标记

数据集目录结构：
workspace/datasets/<name>/
├── dataset.json       # 元数据（任务类型、类别、关键点、划分比例、版本）
├── images/            # 全部图像（classify: images/<class>/ 子目录）
├── labels/            # YOLO 格式标签（classify 无）
├── splits/            # train.txt / val.txt / test.txt（图像路径）
├── ai_marks.json      # AI 生成的标注记录 {image_name: timestamp}
└── snapshots/         # 数据集快照（zip）
"""
import json
import random
import shutil
import time
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict

from core import logger
from core.exceptions import DatasetError
from data.formats import (TASKS, IMAGE_EXTS, read_yolo_file, write_yolo_file,
                          convert_dataset, import_classify_folder, annotations_to_coco)

TASK_LABELS = {
    "classify": "图像分类",
    "detect": "目标检测",
    "segment": "实例分割",
    "pose": "姿态估计",
    "obb": "定向边界框",
}

DEFAULT_PALETTE = [
    "#4f8cff", "#e05c5c", "#4caf7d", "#e6a23c", "#8b5cf6", "#e85aa8", "#14b8c6",
    "#a3a3a3", "#d97706", "#059669", "#7c3aed", "#db2777", "#0ea5e9", "#65a30d",
    "#f43f5e", "#6366f1", "#0891b2", "#ca8a04", "#16a34a", "#9333ea",
]


@dataclass
class ClassInfo:
    id: int
    name: str
    color: str = ""


@dataclass
class KeypointInfo:
    id: int
    name: str


@dataclass
class DatasetMeta:
    name: str
    task: str
    description: str = ""
    classes: List[ClassInfo] = field(default_factory=list)
    keypoints: List[KeypointInfo] = field(default_factory=list)
    skeleton: List[List[int]] = field(default_factory=list)
    split_ratio: Dict[str, float] = field(default_factory=lambda: {"train": 0.7, "val": 0.2, "test": 0.1})
    created_at: str = ""
    updated_at: str = ""
    version: int = 1
    source_format: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetMeta":
        d = dict(d)
        d["classes"] = [ClassInfo(**c) if isinstance(c, dict) else c for c in d.get("classes", [])]
        d["keypoints"] = [KeypointInfo(**k) if isinstance(k, dict) else k for k in d.get("keypoints", [])]
        known = {f: d.get(f) for f in ("name", "task", "description", "skeleton", "split_ratio",
                                       "created_at", "updated_at", "version", "source_format") if f in d}
        return cls(classes=d.pop("classes", []), keypoints=d.pop("keypoints", []), **known)


class Dataset:
    """单个数据集的访问句柄"""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.meta_path = self.root / "dataset.json"
        if not self.meta_path.exists():
            raise DatasetError(f"数据集不存在: {self.root}")
        self.meta = DatasetMeta.from_dict(json.loads(self.meta_path.read_text(encoding="utf-8")))

    # ---------- 路径 ----------
    @property
    def name(self) -> str:
        return self.meta.name

    @property
    def task(self) -> str:
        return self.meta.task

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def labels_dir(self) -> Path:
        return self.root / "labels"

    @property
    def splits_dir(self) -> Path:
        return self.root / "splits"

    @property
    def snapshots_dir(self) -> Path:
        return self.root / "snapshots"

    @property
    def ai_marks_file(self) -> Path:
        return self.root / "ai_marks.json"

    @property
    def class_names(self) -> List[str]:
        return [c.name for c in self.meta.classes]

    def class_by_id(self, cid: int) -> Optional[ClassInfo]:
        for c in self.meta.classes:
            if c.id == cid:
                return c
        return None

    def class_id_by_name(self, name: str) -> Optional[int]:
        for c in self.meta.classes:
            if c.name == name:
                return c.id
        return None

    # ---------- 图像 ----------
    def image_files(self) -> List[Path]:
        """全部图像（支持 images/ 下的子目录，如 train/、val/）"""
        return sorted(p for p in self.images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

    def image_display_name(self, img: Path) -> str:
        """图像在数据集内的相对名（images/ 下的相对路径），
        如 'a.jpg' 或 'train/a.jpg'，用于标注读写等所有接口"""
        return img.relative_to(self.images_dir).as_posix()

    def add_images(self, files: List[Path], progress_cb=None) -> int:
        """复制图像进数据集（去重），返回新增数量"""
        self.images_dir.mkdir(parents=True, exist_ok=True)
        added = 0
        total = len(files)
        for i, f in enumerate(files):
            f = Path(f)
            if not f.exists():
                continue
            dst = self.images_dir / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
                added += 1
            if progress_cb:
                progress_cb(i + 1, total)
        self._touch()
        return added

    def remove_image(self, image_name: str) -> bool:
        img = self.images_dir / image_name
        removed = False
        if img.exists():
            img.unlink()
            removed = True
        lab = self.label_path_for(image_name)
        if lab.exists():
            lab.unlink()
        self.unmark_ai(image_name)
        self._touch()
        return removed

    # ---------- 标注 ----------
    def label_path_for(self, image_name: str) -> Path:
        return self.labels_dir / Path(image_name).with_suffix(".txt")

    def read_annotations(self, image_name: str) -> List[dict]:
        if self.task == "classify":
            cid = self.classify_label(image_name)
            return [{"class_id": cid}] if cid is not None else []
        return read_yolo_file(self.label_path_for(image_name), self.task)

    def write_annotations(self, image_name: str, annotations: List[dict]):
        if self.task == "classify":
            raise DatasetError("分类任务请使用 set_classify_label")
        write_yolo_file(self.label_path_for(image_name), annotations, self.task)
        self._touch()

    def classify_label(self, image_name: str) -> Optional[int]:
        """分类任务：图像所在类别目录 → 类别 id（支持 train/<class>/x.jpg 划分结构）"""
        rel = Path(image_name)
        parts = rel.parts
        if len(parts) >= 2 and parts[0] in ("train", "val", "test"):
            cls_name = parts[1]
        elif rel.parent != Path("."):
            cls_name = parts[0]
        else:
            img = self.images_dir / image_name
            if not img.exists():
                return None
            cls_name = img.parent.name
        return self.class_id_by_name(cls_name)

    def set_classify_label(self, image_name: str, class_name: str):
        """分类任务：把图像移动到对应类别子目录（保留 train/val 划分结构）"""
        img = self.images_dir / image_name
        if not img.exists():
            raise DatasetError(f"图像不存在: {image_name}")
        parts = img.relative_to(self.images_dir).parts
        if parts[0] in ("train", "val", "test"):
            dst_dir = self.images_dir / parts[0] / class_name
        else:
            dst_dir = self.images_dir / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / img.name
        if dst != img:
            if dst.exists():
                dst = dst_dir / f"{img.stem}_{int(time.time())}{img.suffix}"
            shutil.move(str(img), str(dst))
        self._touch()

    # ---------- AI 标注标记 ----------
    def ai_marked(self) -> Dict[str, str]:
        if self.ai_marks_file.exists():
            try:
                return json.loads(self.ai_marks_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def is_ai(self, image_name: str) -> bool:
        return image_name in self.ai_marked()

    def mark_ai(self, image_name: str):
        marks = self.ai_marked()
        marks[image_name] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.ai_marks_file.write_text(json.dumps(marks, ensure_ascii=False, indent=2), encoding="utf-8")

    def unmark_ai(self, image_name: str):
        marks = self.ai_marked()
        if image_name in marks:
            del marks[image_name]
            self.ai_marks_file.write_text(json.dumps(marks, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 划分 ----------
    def generate_splits(self, ratio: Optional[Dict[str, float]] = None, seed: int = 42) -> Dict[str, int]:
        """生成 train/val/test.txt（图像绝对路径），返回各集合图像数"""
        ratio = ratio or self.meta.split_ratio
        if self.task == "classify":
            raise DatasetError("分类任务暂不支持数据划分（直接使用全部图像训练）")
        images = self.image_files()
        if not images:
            raise DatasetError("数据集为空，无法划分")
        rng = random.Random(seed)
        shuffled = images[:]
        rng.shuffle(shuffled)
        train_r = ratio.get("train", 0.7)
        val_r = ratio.get("val", 0.2)
        n_train = int(len(shuffled) * train_r)
        n_val = int(len(shuffled) * val_r)
        splits = {"train": shuffled[:n_train], "val": shuffled[n_train:n_train + n_val], "test": shuffled[n_train + n_val:]}
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        counts = {}
        for key, imgs in splits.items():
            if not imgs:
                continue
            (self.splits_dir / f"{key}.txt").write_text(
                "\n".join(p.as_posix() for p in imgs) + "\n", encoding="utf-8")
            counts[key] = len(imgs)
        self.meta.split_ratio = ratio
        self._touch()
        return counts

    def training_data_path(self) -> str:
        """返回可直接传给 ultralytics 的 data 参数：
        分类任务为图像目录（ultralytics 要求目录，自动使用其中 train/val 子目录），
        其余任务为生成的 data.yaml 路径。"""
        if self.task == "classify":
            return str(self.images_dir)
        return str(self.build_yaml())

    def build_yaml(self, split_names=("train", "val")) -> Path:
        """生成 ultralytics 训练用 data.yaml，返回路径"""
        lines = [f"path: {self.root.as_posix()}"]
        for sn in split_names:
            sf = self.splits_dir / f"{sn}.txt"
            if sf.exists():
                lines.append(f"{sn}: {sf.as_posix()}")
        if self.task == "classify":
            # 若保留了 train/val 划分结构则分别使用，否则全部图像既作训练又作验证
            train_dir = self.images_dir / "train"
            val_dir = self.images_dir / "val"
            if train_dir.is_dir() and val_dir.is_dir():
                lines = [f"path: {self.root.as_posix()}", f"train: {train_dir.as_posix()}",
                         f"val: {val_dir.as_posix()}"]
            else:
                lines = [f"path: {self.root.as_posix()}", f"train: {self.images_dir.as_posix()}",
                         f"val: {self.images_dir.as_posix()}"]
        names = ", ".join(f"{c.id}: '{c.name}'" for c in self.meta.classes)
        lines.append("names: {" + names + "}")
        if self.task == "pose" and self.meta.keypoints:
            lines.append(f"kpt_shape: [{len(self.meta.keypoints)}, 3]")
        yaml_path = self.root / "data.yaml"
        yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return yaml_path

    # ---------- 统计 ----------
    def stats(self) -> dict:
        images = self.image_files()
        per_class: Dict[int, int] = {}
        labeled = 0
        for img in images:
            anns = self.read_annotations(self.image_display_name(img))
            if anns:
                labeled += 1
            for a in anns:
                cid = a["class_id"]
                per_class[cid] = per_class.get(cid, 0) + 1
        return {
            "total": len(images),
            "labeled": labeled,
            "unlabeled": len(images) - labeled,
            "per_class": per_class,
            "ai_marked": len(self.ai_marked()),
            "version": self.meta.version,
            "created_at": self.meta.created_at,
            "updated_at": self.meta.updated_at,
        }

    # ---------- 快照 ----------
    def snapshot(self, note: str = "") -> Path:
        """把 dataset.json + labels/ + ai_marks.json 打包为 zip 快照"""
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        zip_path = self.snapshots_dir / f"snapshot_{ts}_{self.meta.version}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("dataset.json", self.meta_path.read_text(encoding="utf-8"))
            zf.writestr("note.txt", note)
            if self.ai_marks_file.exists():
                zf.write(self.ai_marks_file, "ai_marks.json")
            if self.labels_dir.exists():
                for f in self.labels_dir.iterdir():
                    zf.write(f, f"labels/{f.name}")
        logger.get_logger().info(f"数据集 {self.name} 快照已创建: {zip_path.name}")
        return zip_path

    def restore_snapshot(self, zip_path: Path):
        """恢复快照：还原 dataset.json、labels/、ai_marks.json"""
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.root)
        # 重新加载元数据
        self.meta = DatasetMeta.from_dict(json.loads(self.meta_path.read_text(encoding="utf-8")))
        logger.get_logger().info(f"数据集 {self.name} 已从快照恢复: {Path(zip_path).name}")

    def list_snapshots(self) -> List[Path]:
        if not self.snapshots_dir.exists():
            return []
        return sorted(self.snapshots_dir.glob("*.zip"), reverse=True)

    def _touch(self):
        self.meta.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.meta_path.write_text(json.dumps(self.meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


class DatasetManager:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------- 基础 CRUD ----------
    def list(self) -> List[Dataset]:
        out = []
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and (d / "dataset.json").exists():
                try:
                    out.append(Dataset(d))
                except (DatasetError, json.JSONDecodeError):
                    continue
        return out

    def get(self, name: str) -> Dataset:
        ds = Dataset(self.root / name)
        return ds

    def exists(self, name: str) -> bool:
        return (self.root / name / "dataset.json").exists()

    def create(self, name: str, task: str, classes: List[str], description: str = "",
               keypoints: Optional[List[str]] = None, skeleton: Optional[List[List[int]]] = None,
               split_ratio: Optional[Dict[str, float]] = None) -> Dataset:
        name = name.strip()
        if not name:
            raise DatasetError("数据集名称不能为空")
        if any(ch in name for ch in '\\/:*?"<>|'):
            raise DatasetError("数据集名称含非法字符")
        if self.exists(name):
            raise DatasetError(f"数据集 {name} 已存在")
        if task not in TASKS:
            raise DatasetError(f"不支持的任务类型: {task}")
        if not classes:
            raise DatasetError("至少需要 1 个类别")
        if task == "pose":
            if not keypoints:
                raise DatasetError("姿态估计任务需要配置关键点")
            if not skeleton:
                skeleton = []
        root = self.root / name
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "labels").mkdir(parents=True, exist_ok=True)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        meta = DatasetMeta(
            name=name, task=task, description=description,
            classes=[ClassInfo(id=i, name=c, color=DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)]) for i, c in enumerate(classes)],
            keypoints=[KeypointInfo(id=i, name=k) for i, k in enumerate(keypoints or [])],
            skeleton=skeleton or [],
            split_ratio=split_ratio or {"train": 0.7, "val": 0.2, "test": 0.1},
            created_at=now, updated_at=now, version=1,
        )
        (root / "dataset.json").write_text(json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        logger.get_logger().info(f"数据集创建成功: {name} ({TASK_LABELS.get(task, task)})")
        return Dataset(root)

    def delete(self, name: str):
        root = self.root / name
        if not root.exists():
            raise DatasetError(f"数据集 {name} 不存在")
        shutil.rmtree(root)
        logger.get_logger().info(f"数据集已删除: {name}")

    def rename(self, old: str, new: str):
        ds = self.get(old)
        new = new.strip()
        if self.exists(new):
            raise DatasetError(f"数据集 {new} 已存在")
        ds.meta.name = new
        ds._touch()
        (self.root / old).rename(self.root / new)
        logger.get_logger().info(f"数据集重命名: {old} → {new}")

    # ---------- 导入 / 导出 ----------
    def import_from(self, src: Path, src_format: str, name: str, task: str,
                    class_names: Optional[List[str]] = None, split_ratio=None, progress_cb=None) -> Dataset:
        """导入外部数据集并创建平台数据集"""
        if self.exists(name):
            raise DatasetError(f"数据集 {name} 已存在")
        root = self.root / name
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "labels").mkdir(parents=True, exist_ok=True)
        if src_format == "folder" or (task == "classify" and src_format in ("folder", "")):
            classes = import_classify_folder(src, root / "images", progress_cb)
            return self._finalize_import(name, task, classes, src_format, split_ratio)
        info = convert_dataset(src, src_format, task, root / "images", root / "labels",
                               class_names=class_names, progress_cb=progress_cb)
        return self._finalize_import(name, task, info["class_names"], src_format, split_ratio,
                                     kpt_count=info["kpt_count"])

    def _finalize_import(self, name: str, task: str, classes: List[str], src_format: str,
                         split_ratio, kpt_count: Optional[int] = None) -> Dataset:
        keypoints = None
        skeleton = None
        if task == "pose":
            keypoints = [f"keypoint_{i}" for i in range(kpt_count or 17)]
        ds = self.create(name, task, classes or ["class_0"], split_ratio=split_ratio)
        ds.meta.source_format = src_format
        ds._touch()
        return ds

    def export(self, ds: Dataset, target_format: str, out_dir: Path) -> Path:
        """导出数据集到指定格式，返回输出路径"""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        target_format = target_format.lower()
        if target_format == "yolo":
            dst = out_dir / ds.name
            (dst / "images").mkdir(parents=True, exist_ok=True)
            (dst / "labels").mkdir(parents=True, exist_ok=True)
            for img in ds.image_files():
                rel = img.relative_to(ds.images_dir)
                dst_img = dst / "images" / rel
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img, dst_img)
            if ds.task != "classify":
                for lab in ds.labels_dir.rglob("*.txt"):
                    rel = lab.relative_to(ds.labels_dir)
                    dst_lab = dst / "labels" / rel
                    dst_lab.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(lab, dst_lab)
            (dst / "classes.txt").write_text("\n".join(ds.class_names) + "\n", encoding="utf-8")
            return dst
        if target_format == "coco":
            if ds.task == "obb":
                raise DatasetError("COCO 格式不支持 OBB 任务")
            if ds.task == "classify":
                raise DatasetError("COCO 格式不支持分类任务")
            file_ann = {}
            for img in ds.image_files():
                name = ds.image_display_name(img)
                anns = ds.read_annotations(name)
                if anns:
                    file_ann[name] = anns
            coco = annotations_to_coco(file_ann, ds.class_names, ds.task, ds.image_files())
            out_json = out_dir / f"{ds.name}.json"
            out_json.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
            return out_json
        if target_format in ("voc", "pascal_voc"):
            if ds.task != "detect":
                raise DatasetError("VOC 格式仅支持目标检测任务")
            dst = out_dir / ds.name
            (dst / "Annotations").mkdir(parents=True, exist_ok=True)
            (dst / "JPEGImages").mkdir(parents=True, exist_ok=True)
            from data.formats import write_voc_xml
            from PIL import Image
            for img in ds.image_files():
                name = ds.image_display_name(img)
                flat_name = name.replace("/", "_")
                anns = ds.read_annotations(name)
                shutil.copy2(img, dst / "JPEGImages" / flat_name)
                with Image.open(img) as im:
                    w, h = im.size
                write_voc_xml(dst / "Annotations" / (Path(flat_name).stem + ".xml"),
                              flat_name, w, h, anns, ds.class_names)
            return dst
        raise DatasetError(f"不支持的导出格式: {target_format}")
