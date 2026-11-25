# -*- coding: utf-8 -*-
"""离线数据增强：翻转 / 旋转 / HSV 抖动 / 亮度对比度 / 高斯噪声 / Mosaic

训练时的在线增强（mosaic/mixup/颜色抖动等）由 ultralytics 自动处理；
本模块提供离线增强，用于扩充数据集文件本身。
"""
import random
from pathlib import Path
from typing import List

import cv2
import numpy as np

from core.exceptions import DatasetError
from data.dataset_manager import Dataset

STRATEGIES = {
    "flip_h": "水平翻转",
    "flip_v": "垂直翻转",
    "rotate90": "旋转90°",
    "hsv": "HSV 颜色抖动",
    "contrast": "亮度/对比度调整",
    "noise": "高斯噪声",
    "mosaic": "Mosaic 拼接（仅检测）",
}


def _swap_pairs(keypoint_names: List[str]) -> dict:
    """根据关键点名称自动识别 left/right 对称对，返回交换索引映射"""
    swap = {}
    for i, name in enumerate(keypoint_names):
        if name.startswith("left_"):
            j = next((k for k, n in enumerate(keypoint_names) if n == "right_" + name[5:]), None)
            if j is not None:
                swap[i] = j
        elif name.startswith("right_"):
            j = next((k for k, n in enumerate(keypoint_names) if n == "left_" + name[6:]), None)
            if j is not None:
                swap[i] = j
        elif name.startswith("L"):
            j = next((k for k, n in enumerate(keypoint_names) if n == "R" + name[1:]), None)
            if j is not None:
                swap[i] = j
    return swap


def _transform_annotation(ann: dict, task: str, tx, swap: dict = None) -> dict:
    """对标注做坐标变换 tx(x, y) -> (x', y')；角度变换由调用方处理"""
    import copy
    a = copy.deepcopy(ann)
    if task == "detect":
        cx, cy, w, h = a["bbox"]
        ncx, ncy = tx(cx, cy)
        a["bbox"] = [ncx, ncy, w, h]
    elif task == "segment":
        a["points"] = [tx(x, y) for x, y in a["points"]]
        xs = [p[0] for p in a["points"]]
        ys = [p[1] for p in a["points"]]
        a["bbox"] = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
    elif task == "pose":
        cx, cy, w, h = a["bbox"]
        ncx, ncy = tx(cx, cy)
        a["bbox"] = [ncx, ncy, w, h]
        kps = []
        for i, (x, y, v) in enumerate(a["keypoints"]):
            nx, ny = tx(x, y)
            kps.append([nx, ny, v])
        if swap:
            kps2 = [k[:] for k in kps]
            for i, j in swap.items():
                if i < len(kps) and j < len(kps):
                    kps2[i], kps2[j] = kps[j], kps[i]
            kps = kps2
        a["keypoints"] = kps
    elif task == "obb":
        cx, cy, w, h = a["bbox"]
        ncx, ncy = tx(cx, cy)
        a["bbox"] = [ncx, ncy, w, h]
    return a


class Augmenter:
    def __init__(self, ds: Dataset):
        if ds.task == "classify":
            raise DatasetError("分类任务不支持离线增强")
        self.ds = ds

    def apply(self, strategy: str, params: dict = None, suffix: str = None, progress_cb=None) -> int:
        """对整个数据集应用增强，生成新的图像+标签文件，返回生成数量"""
        params = params or {}
        suffix = suffix or strategy
        if strategy not in STRATEGIES:
            raise DatasetError(f"未知增强策略: {strategy}")
        images = self.ds.image_files()
        labeled = [img for img in images if self.ds.read_annotations(img.name)]
        if strategy in ("hsv", "contrast", "noise"):
            targets = images  # 纯图像变换，可应用于未标注图像
        else:
            targets = labeled
        swap = _swap_pairs([k.name for k in self.ds.meta.keypoints]) if self.ds.task == "pose" else None
        count = 0
        total = len(targets)
        for i, img in enumerate(targets):
            try:
                self._apply_one(img, strategy, params, suffix, swap)
                count += 1
            except Exception:
                continue
            if progress_cb:
                progress_cb(i + 1, total)
        return count

    def _apply_one(self, img: Path, strategy: str, params: dict, suffix: str, swap: dict):
        image = cv2.imread(str(img))
        if image is None:
            return
        h, w = image.shape[:2]
        anns = self.ds.read_annotations(img.name)
        new_image = image
        new_anns = anns

        if strategy == "flip_h":
            new_image = cv2.flip(image, 1)
            tx = lambda x, y: (1 - x, y)
            new_anns = [_transform_annotation(a, self.ds.task, tx, swap) for a in anns]
            if self.ds.task == "obb":
                for a in new_anns:
                    a["angle"] = (180 - a["angle"]) % 180
        elif strategy == "flip_v":
            new_image = cv2.flip(image, 0)
            tx = lambda x, y: (x, 1 - y)
            new_anns = [_transform_annotation(a, self.ds.task, tx, None) for a in anns]
            if self.ds.task == "obb":
                for a in new_anns:
                    a["angle"] = (-a["angle"]) % 180
        elif strategy == "rotate90":
            new_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            tx = lambda x, y: (1 - y, x)
            new_anns = [_transform_annotation(a, self.ds.task, tx, None) for a in anns]
            if self.ds.task == "obb":
                for a in new_anns:
                    a["angle"] = (a["angle"] + 90) % 180
        elif strategy == "hsv":
            dh = params.get("hue", 15)
            ds_ = params.get("sat", 0.3)
            dv = params.get("val", 0.3)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-dh, dh)) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(1 - ds_, 1 + ds_), 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(1 - dv, 1 + dv), 0, 255)
            new_image = hsv.astype(np.uint8)
            new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2BGR)
        elif strategy == "contrast":
            alpha = params.get("alpha", random.uniform(0.6, 1.4))
            beta = params.get("beta", random.uniform(-30, 30))
            new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        elif strategy == "noise":
            sigma = params.get("sigma", 15)
            noise = np.random.normal(0, sigma, image.shape).astype(np.int16)
            new_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif strategy == "mosaic":
            new_image, new_anns = self._mosaic(img, params.get("size", 640))
        else:
            return

        stem = f"{img.stem}_{suffix}"
        new_img_path = self.ds.images_dir / f"{stem}{img.suffix}"
        cv2.imwrite(str(new_img_path), new_image)
        if new_anns:
            self.ds.write_annotations(f"{stem}{img.suffix}", new_anns)

    def _mosaic(self, img: Path, size: int = 640):
        """4 图 Mosaic（仅检测任务，其他任务抛错）"""
        if self.ds.task != "detect":
            raise DatasetError("Mosaic 仅支持目标检测任务")
        labeled = [p for p in self.ds.image_files() if self.ds.read_annotations(p.name)]
        if len(labeled) < 4:
            raise DatasetError("Mosaic 需要至少 4 张带标注的图像")
        picks = [img] + random.sample([p for p in labeled if p != img], 3)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        anns_out = []
        half = size // 2
        for k, p in enumerate(picks):
            im = cv2.imread(str(p))
            if im is None:
                continue
            im = cv2.resize(im, (half, half))
            row, col = k // 2, k % 2
            y0, x0 = row * half, col * half
            canvas[y0:y0 + half, x0:x0 + half] = im
            for a in self.ds.read_annotations(p.name):
                cx, cy, w_, h_ = a["bbox"]
                ncx = (cx / 2) + (col / 2)
                ncy = (cy / 2) + (row / 2)
                # 丢弃中心落在块外或过大的框
                if ncx < col / 2 or ncx > (col + 1) / 2 or ncy < row / 2 or ncy > (row + 1) / 2:
                    continue
                anns_out.append({"class_id": a["class_id"],
                                 "bbox": [ncx, ncy, max(0.01, w_ / 2), max(0.01, h_ / 2)]})
        return canvas, anns_out
