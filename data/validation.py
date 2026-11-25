# -*- coding: utf-8 -*-
"""数据质量验证：图像完整性、标注合法性、类别一致性"""
from typing import List

from data.dataset_manager import Dataset

ISSUE_IMAGE_CORRUPT = "图像损坏"
ISSUE_LABEL_PARSE = "标注解析失败"
ISSUE_BBOX_OUT_OF_RANGE = "标注越界"
ISSUE_CLASS_ID = "类别ID非法"
ISSUE_KEYPOINT_COUNT = "关键点数量不符"
ISSUE_POLYGON = "多边形非法"
ISSUE_OBB_ANGLE = "旋转框角度非法"
ISSUE_MISSING_LABEL = "缺少标注"


def _check_bounds(values, tol=0.01):
    for v in values:
        if v < -tol or v > 1 + tol:
            return False
    return True


def _check_image_annotations(ds: Dataset, img, expected_kpts: int) -> List[dict]:
    issues = []
    img_name = ds.image_display_name(img)

    # 图像完整性
    try:
        from PIL import Image
        with Image.open(img) as im:
            im.verify()
    except Exception:
        issues.append({"image": img_name, "type": ISSUE_IMAGE_CORRUPT, "message": "图像无法打开或已损坏"})
        return issues

    if ds.task == "classify":
        cid = ds.classify_label(img_name)
        if cid is None:
            issues.append({"image": img_name, "type": ISSUE_CLASS_ID, "message": "图像不在任何已知类别目录中"})
        return issues

    lab = ds.label_path_for(img_name)
    if not lab.exists():
        issues.append({"image": img_name, "type": ISSUE_MISSING_LABEL, "message": "无标注文件"})
        return issues

    try:
        anns = ds.read_annotations(img_name)
    except Exception as e:
        issues.append({"image": img_name, "type": ISSUE_LABEL_PARSE, "message": str(e)})
        return issues

    n_classes = len(ds.meta.classes)
    for ann in anns:
        cid = ann.get("class_id", -1)
        if cid < 0 or cid >= n_classes:
            issues.append({"image": img_name, "type": ISSUE_CLASS_ID, "message": f"类别ID {cid} 超出范围"})
        bbox = ann.get("bbox")
        if bbox is not None:
            cx, cy, w, h = bbox[:4]
            if not _check_bounds([cx, cy]) or w <= 0 or h <= 0 or cx - w / 2 < -0.01 or cx + w / 2 > 1.01 \
                    or cy - h / 2 < -0.01 or cy + h / 2 > 1.01:
                issues.append({"image": img_name, "type": ISSUE_BBOX_OUT_OF_RANGE, "message": f"边界框越界: {bbox}"})
        if ds.task == "segment":
            pts = ann.get("points", [])
            if len(pts) < 3:
                issues.append({"image": img_name, "type": ISSUE_POLYGON, "message": "多边形少于 3 个顶点"})
            elif not _check_bounds([c for p in pts for c in p]):
                issues.append({"image": img_name, "type": ISSUE_BBOX_OUT_OF_RANGE, "message": "多边形顶点越界"})
        if ds.task == "pose":
            kps = ann.get("keypoints", [])
            if expected_kpts and len(kps) != expected_kpts:
                issues.append({"image": img_name, "type": ISSUE_KEYPOINT_COUNT,
                               "message": f"关键点数 {len(kps)} ≠ 配置数 {expected_kpts}"})
            if not _check_bounds([c for x, y, v in kps for c in (x, y)]):
                issues.append({"image": img_name, "type": ISSUE_BBOX_OUT_OF_RANGE, "message": "关键点越界"})
        if ds.task == "obb":
            angle = ann.get("angle", 0.0)
            if angle < -180 or angle > 180:
                issues.append({"image": img_name, "type": ISSUE_OBB_ANGLE, "message": f"角度非法: {angle}"})
    return issues


def validate_dataset(ds: Dataset, progress_cb=None) -> dict:
    """返回验证报告 dict"""
    images = ds.image_files()
    expected_kpts = len(ds.meta.keypoints) if ds.task == "pose" else 0
    issues: List[dict] = []
    corrupt = set()
    per_class = {c.id: 0 for c in ds.meta.classes}
    total = len(images)

    for i, img in enumerate(images):
        img_name = ds.image_display_name(img)
        img_issues = _check_image_annotations(ds, img, expected_kpts)
        if any(x["type"] == ISSUE_IMAGE_CORRUPT for x in img_issues):
            corrupt.add(img_name)
        else:
            anns = ds.read_annotations(img_name)
            for a in anns:
                cid = a["class_id"]
                if cid in per_class:
                    per_class[cid] += 1
        issues.extend(img_issues)
        if progress_cb:
            progress_cb(i + 1, total)

    bad = len(corrupt) + sum(1 for x in issues if x["type"] == ISSUE_LABEL_PARSE)
    return {
        "task": ds.task,
        "task_label": ds.task,
        "total": total,
        "valid": total - bad,
        "invalid": bad,
        "per_class": per_class,
        "class_names": ds.class_names,
        "ai_marked": len(ds.ai_marked()),
        "issues": issues,
    }
