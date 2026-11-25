# -*- coding: utf-8 -*-
"""冒烟测试：验证核心模块、认证、数据层、服务层的非 GUI 逻辑。

运行：python tests/test_smoke.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
import numpy as np


def make_image(path: Path, size=(64, 64), color=(128, 128, 128)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((*size, 3), color, dtype=np.uint8)).save(path)


def test_config():
    from core.config import ConfigManager
    cm = ConfigManager(Path(tempfile.mkdtemp()))
    cm.load()
    assert cm.config.training.epochs == 100
    cm.update_section("training", {"epochs": 50})
    assert cm.config.training.epochs == 50
    print("✓ 配置管理")


def test_auth():
    from auth.manager import AuthManager
    am = AuthManager(Path(tempfile.mkdtemp()))
    assert am.ensure_default_admin()
    user, msg = am.login("admin", "admin123")
    assert user is not None, msg
    assert user.is_admin
    assert am.is_authenticated()
    ok, msg = am.add_user("tester", "123456")
    assert ok, msg
    user2, msg = am.login("tester", "wrong")
    assert user2 is None
    am.logout()
    assert not am.is_authenticated()
    print("✓ 认证管理")


def test_dataset_and_formats():
    from data.dataset_manager import DatasetManager
    from data.formats import parse_yolo_line, format_yolo_line

    root = Path(tempfile.mkdtemp())
    dm = DatasetManager(root / "datasets")
    ds = dm.create("test_detect", "detect", ["person", "car"])
    assert ds.stats()["total"] == 0

    # YOLO 标签往返
    line = "0 0.5 0.5 0.4 0.3"
    ann = parse_yolo_line(line, "detect")
    assert format_yolo_line(ann, "detect").startswith("0 ")
    ds.write_annotations("a.jpg", [ann])
    assert ds.read_annotations("a.jpg")[0]["class_id"] == 0

    # 图像与划分
    img = root / "tmp.jpg"
    make_image(img)
    ds.add_images([img])
    make_image(root / "b.jpg")
    ds.add_images([root / "b.jpg"])
    counts = ds.generate_splits()
    assert sum(counts.values()) == 2
    yaml_path = ds.build_yaml()
    assert yaml_path.exists()
    print("✓ 数据集管理 + YOLO 标签")

    # 姿态数据集
    ds_pose = dm.create("test_pose", "pose", ["person"], keypoints=["nose", "left_eye", "right_eye"])
    pose_line = "0 0.5 0.5 0.4 0.3 0.5 0.4 2 0.45 0.4 2 0.55 0.4 2"
    ds_pose.write_annotations("p.jpg", [parse_yolo_line(pose_line, "pose")])
    assert len(ds_pose.read_annotations("p.jpg")[0]["keypoints"]) == 3
    print("✓ 姿态数据集")

    # 导出 COCO / VOC
    out = root / "out"
    coco_json = dm.export(ds, "coco", out)
    assert coco_json.exists()
    voc_dir = dm.export(ds, "voc", out)
    assert (voc_dir / "Annotations").exists()
    print("✓ COCO / VOC 导出")

    # 快照
    snap = ds.snapshot("test")
    assert snap.exists()
    ds2 = dm.get("test_detect")
    ds2.restore_snapshot(snap)
    print("✓ 数据集快照")


def test_validation():
    from data.dataset_manager import DatasetManager
    from data.validation import validate_dataset

    root = Path(tempfile.mkdtemp())
    dm = DatasetManager(root / "datasets")
    ds = dm.create("val_test", "detect", ["person"])
    img = root / "ok.jpg"
    make_image(img)
    ds.add_images([img])
    ds.write_annotations("ok.jpg", [{"class_id": 0, "bbox": [0.5, 0.5, 0.4, 0.3]}])
    # 越界标注
    img2 = root / "bad.jpg"
    make_image(img2)
    ds.add_images([img2])
    ds.write_annotations("bad.jpg", [{"class_id": 0, "bbox": [1.5, 0.5, 0.4, 0.3]}])
    report = validate_dataset(ds)
    assert report["total"] == 2
    assert any(x["type"] == "标注越界" for x in report["issues"])
    print("✓ 数据验证")


def test_augmentation():
    from data.dataset_manager import DatasetManager
    from data.augmentation import Augmenter

    root = Path(tempfile.mkdtemp())
    dm = DatasetManager(root / "datasets")
    ds = dm.create("aug_test", "detect", ["person"])
    img = root / "a.jpg"
    make_image(img)
    ds.add_images([img])
    ds.write_annotations("a.jpg", [{"class_id": 0, "bbox": [0.5, 0.5, 0.4, 0.3]}])
    n = Augmenter(ds).apply("flip_h")
    assert n >= 1
    assert len(ds.image_files()) >= 2
    print("✓ 离线数据增强")


def test_dependencies():
    from core.dependencies import check_dependencies, resolve_device, is_full_install
    deps = check_dependencies()
    assert deps["python"]
    print(f"✓ 依赖检测: python={deps['python']}, torch={deps['PyTorch']}, device={resolve_device('auto')}")


if __name__ == "__main__":
    test_config()
    test_auth()
    test_dataset_and_formats()
    test_validation()
    test_augmentation()
    test_dependencies()
    print("\n全部冒烟测试通过 ✅")
