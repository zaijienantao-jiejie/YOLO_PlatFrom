# -*- coding: utf-8 -*-
"""数据集格式解析与转换：YOLO / COCO / Pascal VOC / 文件夹分类

内部标注格式（统一数据模型）：
- detect : {"class_id": int, "bbox": [cx, cy, w, h]}       归一化
- segment: {"class_id": int, "points": [(x, y), ...]}      归一化多边形
- pose   : {"class_id": int, "bbox": [cx, cy, w, h],
            "keypoints": [(x, y, v), ...]}                  v: 0不可见 1遮挡 2可见
- obb    : {"class_id": int, "bbox": [cx, cy, w, h], "angle": deg}
- classify: 无标签文件，图像按类别存于 images/<class>/ 子目录
"""
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

from core.exceptions import DatasetError

TASKS = ("classify", "detect", "segment", "pose", "obb")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ================================================================ YOLO 标签
def parse_yolo_line(line: str, task: str) -> dict:
    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError(f"标注行格式错误: {line!r}")
    cls = int(parts[0])
    rest = [float(x) for x in parts[1:]]
    if task == "detect":
        if len(rest) < 4:
            raise ValueError("检测标注缺少 bbox")
        return {"class_id": cls, "bbox": rest[:4]}
    if task == "segment":
        if len(rest) < 6 or len(rest) % 2:
            raise ValueError("分割标注点数量错误")
        return {"class_id": cls, "points": [(rest[i], rest[i + 1]) for i in range(0, len(rest), 2)]}
    if task == "pose":
        if len(rest) < 7 or (len(rest) - 4) % 3:
            raise ValueError("姿态标注格式错误")
        return {"class_id": cls, "bbox": rest[:4],
                "keypoints": [(rest[i], rest[i + 1], int(rest[i + 2])) for i in range(4, len(rest), 3)]}
    if task == "obb":
        if len(rest) < 5:
            raise ValueError("OBB 标注缺少角度")
        return {"class_id": cls, "bbox": rest[:4], "angle": rest[4]}
    raise DatasetError(f"不支持的任务类型: {task}")


def format_yolo_line(ann: dict, task: str) -> str:
    cls = ann["class_id"]
    if task == "detect":
        return f"{cls} " + " ".join(f"{x:.6g}" for x in ann["bbox"])
    if task == "segment":
        pts = " ".join(f"{x:.6g}" for p in ann["points"] for x in p)
        return f"{cls} {pts}"
    if task == "pose":
        bbox = " ".join(f"{x:.6g}" for x in ann["bbox"])
        kps = " ".join(f"{x:.6g} {y:.6g} {int(v)}" for x, y, v in ann["keypoints"])
        return f"{cls} {bbox} {kps}"
    if task == "obb":
        return f"{cls} " + " ".join(f"{x:.6g}" for x in ann["bbox"]) + f" {ann['angle']:.6g}"
    raise DatasetError(f"不支持的任务类型: {task}")


def read_yolo_file(path: Path, task: str) -> List[dict]:
    if not path.exists():
        return []
    anns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                anns.append(parse_yolo_line(line, task))
            except ValueError:
                continue
    return anns


def write_yolo_file(path: Path, annotations: List[dict], task: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(format_yolo_line(a, task) for a in annotations)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


# ================================================================ COCO
def _bbox_from_coco(bbox, w, h):
    """xywh 像素 → cxcywh 归一化"""
    x, y, bw, bh = bbox
    return [x / w + bw / (2 * w), y / h + bh / (2 * h), bw / w, bh / h]


def coco_to_annotations(coco: dict, task: str) -> Dict[str, List[dict]]:
    """COCO JSON → {file_name: [annotation, ...]}（category_id 重映射为 0 起始索引）"""
    img_info = {im["id"]: im for im in coco.get("images", [])}
    cats = coco.get("categories", [])
    id2idx = {c["id"]: i for i, c in enumerate(cats)}
    out: Dict[str, List[dict]] = {}
    for ann in coco.get("annotations", []):
        im = img_info.get(ann["image_id"])
        if im is None:
            continue
        w, h = im["width"], im["height"]
        item = {"class_id": id2idx.get(ann["category_id"], 0)}
        if task in ("detect", "segment", "pose", "obb"):
            item["bbox"] = _bbox_from_coco(ann["bbox"], w, h)
        if task == "segment":
            seg = ann.get("segmentation", [])
            if not seg:
                continue
            poly = seg[0]
            item["points"] = [(poly[i] / w, poly[i + 1] / h) for i in range(0, len(poly), 2)]
        if task == "pose":
            kps = ann.get("keypoints", [])
            item["keypoints"] = [(kps[i] / w, kps[i + 1] / h, int(kps[i + 2])) for i in range(0, len(kps), 3)]
        out.setdefault(im["file_name"], []).append(item)
    return out


def annotations_to_coco(file_ann_map: Dict[str, List[dict]], class_names: List[str], task: str,
                        image_paths: List[Path]) -> dict:
    """内部标注 → COCO JSON（支持 detect/segment/pose）"""
    coco = {
        "info": {"description": "Exported by YOLO Platform"},
        "images": [], "annotations": [],
        "categories": [{"id": i, "name": n} for i, n in enumerate(class_names)],
    }
    by_name = {p.name: p for p in image_paths}
    ann_id = 0
    for img_id, (fname, anns) in enumerate(sorted(file_ann_map.items())):
        img = by_name.get(Path(fname).name)
        if img is None:
            continue
        from PIL import Image
        try:
            with Image.open(img) as im:
                w, h = im.size
        except Exception:
            continue
        coco["images"].append({"id": img_id, "file_name": img.name, "width": w, "height": h})
        for a in anns:
            bbox = a["bbox"]
            x = (bbox[0] - bbox[2] / 2) * w
            y = (bbox[1] - bbox[3] / 2) * h
            bw = bbox[2] * w
            bh = bbox[3] * h
            item = {
                "id": ann_id, "image_id": img_id, "category_id": a["class_id"],
                "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0,
            }
            if task == "segment":
                item["segmentation"] = [[c for p in a["points"] for c in (p[0] * w, p[1] * h)]]
            if task == "pose":
                item["keypoints"] = [c for x, y, v in a["keypoints"] for c in (x * w, y * h, v)]
                item["num_keypoints"] = sum(1 for _, _, v in a["keypoints"] if v == 2)
            coco["annotations"].append(item)
            ann_id += 1
    return coco


# ================================================================ Pascal VOC
def read_voc_xml(path: Path, class_lookup: Dict[str, int]) -> List[dict]:
    root = ET.parse(path).getroot()
    size = root.find("size")
    w = float(size.find("width").text)
    h = float(size.find("height").text)
    anns = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in class_lookup:
            continue
        b = obj.find("bndbox")
        xmin, ymin = float(b.find("xmin").text), float(b.find("ymin").text)
        xmax, ymax = float(b.find("xmax").text), float(b.find("ymax").text)
        anns.append({
            "class_id": class_lookup[name],
            "bbox": [(xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h, (xmax - xmin) / w, (ymax - ymin) / h],
        })
    return anns


def write_voc_xml(path: Path, image_name: str, w: int, h: int, anns: List[dict], class_names: List[str]):
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "JPEGImages"
    ET.SubElement(root, "filename").text = image_name
    size = ET.SubElement(root, "size")
    for tag, val in (("width", w), ("height", h), ("depth", 3)):
        ET.SubElement(size, tag).text = str(val)
    for a in anns:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = class_names[a["class_id"]]
        ET.SubElement(obj, "difficult").text = "0"
        b = ET.SubElement(obj, "bndbox")
        cx, cy, bw, bh = a["bbox"]
        xmin = max(0, int((cx - bw / 2) * w))
        ymin = max(0, int((cy - bh / 2) * h))
        xmax = min(w, int((cx + bw / 2) * w))
        ymax = min(h, int((cy + bh / 2) * h))
        for tag, val in (("xmin", xmin), ("ymin", ymin), ("xmax", xmax), ("ymax", ymax)):
            ET.SubElement(b, tag).text = str(val)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


# ================================================================ 导入
def _read_classes(src: Path) -> List[str]:
    """从 classes.txt 或 data.yaml 读取类别名"""
    cls_file = src / "classes.txt"
    if cls_file.exists():
        names = [l.strip() for l in cls_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        if names:
            return names
    data_yaml = src / "data.yaml"
    if data_yaml.exists():
        import yaml
        raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
        names = raw.get("names")
        if isinstance(names, dict):
            return [str(names[i]) for i in sorted(names)]
        if isinstance(names, list):
            return [str(n) for n in names]
    return []


def _infer_max_class(src: Path, task: str) -> int:
    lab_dir = src / "labels"
    max_id = -1
    if lab_dir.exists():
        for f in lab_dir.rglob("*.txt"):
            try:
                line = f.read_text(encoding="utf-8").splitlines()[0].strip()
                if line:
                    max_id = max(max_id, int(line.split()[0]))
            except (ValueError, IndexError, OSError):
                continue
    return max_id


def _pairs_from(img_root: Path, lab_root: Path, prefix: str = "") -> List[tuple]:
    """递归收集 (图像路径, 标签路径|None, 数据集内部相对名)"""
    out = []
    for img in sorted(img_root.rglob("*")):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = img.relative_to(img_root)
        rel_name = f"{prefix}/{rel.as_posix()}" if prefix else rel.as_posix()
        lab = lab_root / rel.with_suffix(".txt")
        out.append((img, lab, rel_name))
    return out


def _collect_yolo_pairs(src: Path) -> List[tuple]:
    """收集 YOLO 数据集图像与标签，支持两种布局：
    1) src/images/ + src/labels/（可含 train/val 子目录）
    2) src/train/images/ + src/train/labels/（ultralytics 分目录布局）
    """
    pairs = []
    if (src / "images").is_dir():
        pairs.extend(_pairs_from(src / "images", src / "labels"))
    for sub in sorted(p for p in src.iterdir() if p.is_dir()):
        if sub.name != "images" and (sub / "images").is_dir():
            pairs.extend(_pairs_from(sub / "images", sub / "labels", prefix=sub.name))
    # 按内部相对名去重（两种布局同时存在时优先前者）
    seen = {}
    for img, lab, rel_name in pairs:
        seen.setdefault(rel_name, (img, lab, rel_name))
    return list(seen.values())


def _match_image(base: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        for cand in (base / (stem + ext), base / (stem + ext.upper())):
            if cand.exists():
                return cand
    # 全目录搜索
    for ext in IMAGE_EXTS:
        for cand in base.rglob(f"*{ext}"):
            if cand.stem == stem:
                return cand
    return None


def _collect_voc_classes(ann_dir: Path) -> List[str]:
    names = []
    for xml in sorted(ann_dir.glob("*.xml")):
        try:
            root = ET.parse(xml).getroot()
            for obj in root.findall("object"):
                n = obj.find("name").text
                if n and n not in names:
                    names.append(n)
        except ET.ParseError:
            continue
    return names


def convert_dataset(src: Path, src_format: str, task: str, dst_images: Path, dst_labels: Path,
                    class_names: Optional[List[str]] = None, progress_cb=None) -> dict:
    """外部数据集 → 平台内部格式（images/ + labels/ YOLO 标签）。
    返回 {"class_names": [...], "kpt_count": int|None, "image_count": int}"""
    src = Path(src)
    src_format = (src_format or "").lower()
    dst_images.mkdir(parents=True, exist_ok=True)
    if task != "classify":
        dst_labels.mkdir(parents=True, exist_ok=True)
    kpt_count = None
    count = 0

    if src_format == "yolo":
        pairs = _collect_yolo_pairs(src)
        if not pairs:
            raise DatasetError("未在数据集中找到图像（支持 images/ 子目录及 train/images 分目录布局）")
        if class_names is None:
            class_names = _read_classes(src)
        if not class_names:
            class_names = [f"class_{i}" for i in range(_infer_max_class(src, task) + 1)]
        total = len(pairs)
        for i, (img, lab, rel_name) in enumerate(pairs):
            dst_img = dst_images / rel_name
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            if not dst_img.exists():
                shutil.copy2(img, dst_img)
            if lab is not None and lab.exists():
                dst_lab = dst_labels / Path(rel_name).with_suffix(".txt")
                dst_lab.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(lab, dst_lab)
                if task == "pose" and kpt_count is None:
                    anns = read_yolo_file(lab, task)
                    if anns and anns[0].get("keypoints"):
                        kpt_count = len(anns[0]["keypoints"])
            count += 1
            if progress_cb:
                progress_cb(i + 1, total)
        return {"class_names": class_names, "kpt_count": kpt_count, "image_count": count}

    if src_format == "coco":
        if not src.is_file():
            raise DatasetError("COCO 格式需指定 annotations.json 文件")
        coco = json.loads(src.read_text(encoding="utf-8"))
        cats = coco.get("categories", [])
        if class_names is None:
            class_names = [c["name"] for c in cats]
        file_ann = coco_to_annotations(coco, task)
        base = src.parent
        total = len(file_ann)
        for i, (fname, anns) in enumerate(file_ann.items()):
            img = _match_image(base, Path(fname).stem)
            if img is None:
                continue
            new_img = dst_images / f"{img.stem}{img.suffix.lower()}"
            if not new_img.exists():
                shutil.copy2(img, new_img)
            write_yolo_file(dst_labels / (img.stem + ".txt"), anns, task)
            if task == "pose" and kpt_count is None and anns and anns[0].get("keypoints"):
                kpt_count = len(anns[0]["keypoints"])
            count += 1
            if progress_cb:
                progress_cb(i + 1, total)
        return {"class_names": class_names, "kpt_count": kpt_count, "image_count": count}

    if src_format in ("voc", "pascal_voc"):
        ann_dir = src / "Annotations"
        img_dir = src / "JPEGImages"
        if not ann_dir.exists() or not img_dir.exists():
            raise DatasetError("VOC 数据集需包含 Annotations/ 与 JPEGImages/ 目录")
        if class_names is None:
            class_names = _collect_voc_classes(ann_dir)
        lookup = {n: i for i, n in enumerate(class_names)}
        xmls = sorted(ann_dir.glob("*.xml"))
        for i, xml in enumerate(xmls):
            anns = read_voc_xml(xml, lookup)
            img = _match_image(img_dir, xml.stem)
            if img is None:
                continue
            shutil.copy2(img, dst_images / img.name)
            write_yolo_file(dst_labels / (xml.stem + ".txt"), anns, "detect")
            count += 1
            if progress_cb:
                progress_cb(i + 1, len(xmls))
        return {"class_names": class_names, "kpt_count": None, "image_count": count}

    raise DatasetError(f"不支持的源格式: {src_format}")


def import_classify_folder(src: Path, dst_images: Path, progress_cb=None) -> List[str]:
    """分类数据集导入，支持两种布局：
    1) src/<class>/*.jpg                                （各类别一个子目录）
    2) src/train/<class>/*.jpg + src/val/<class>/...    （ultralytics 分类布局，保留 train/val 划分）
    返回类别名列表（按文件夹名排序）"""
    src = Path(src)
    split_names = ("train", "val", "test")
    split_dirs = [src / n for n in split_names if (src / n).is_dir()]
    has_root_images = any(f.is_file() and f.suffix.lower() in IMAGE_EXTS for f in src.iterdir())
    if split_dirs and not has_root_images:
        # 布局2：保留划分结构 → dst_images/<split>/<class>/xxx.jpg
        classes = set()
        for sd in split_dirs:
            for cd in sd.iterdir():
                if cd.is_dir():
                    classes.add(cd.name)
        total = sum(
            1 for sd in split_dirs
            for cd in sd.iterdir() if cd.is_dir()
            for f in cd.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
        done = 0
        for sd in split_dirs:
            for cls_dir in sorted(d for d in sd.iterdir() if d.is_dir()):
                dst_cls = dst_images / sd.name / cls_dir.name
                dst_cls.mkdir(parents=True, exist_ok=True)
                for f in sorted(p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
                    dst = dst_cls / f.name
                    if not dst.exists():  # train 优先，跨划分同名文件不互相覆盖
                        shutil.copy2(f, dst)
                    done += 1
                    if progress_cb:
                        progress_cb(done, max(total, 1))
        return sorted(classes)
    # 布局1：顶层即类别目录
    classes = sorted(d.name for d in src.iterdir() if d.is_dir())
    for cls in classes:
        (dst_images / cls).mkdir(parents=True, exist_ok=True)
        files = [f for f in (src / cls).iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        for i, f in enumerate(files):
            shutil.copy2(f, dst_images / cls / f.name)
            if progress_cb:
                progress_cb(i + 1, len(files))
    return classes
