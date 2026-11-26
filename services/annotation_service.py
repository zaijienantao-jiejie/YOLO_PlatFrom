# -*- coding: utf-8 -*-
"""AI 辅助标注：用预训练模型对数据集图像生成预标注"""
import shutil
import time
from typing import List, Optional

from PySide6.QtCore import QThread, Signal

from core import logger
from core.dependencies import resolve_device
from data.dataset_manager import Dataset

COCO_NAMES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
              "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
              "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
              "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
              "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

PRETRAINED_MODELS = {
    "detect": ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
               "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
               "yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt"],
    "segment": ["yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt",
                "yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt"],
    "pose": ["yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt", "yolo11l-pose.pt", "yolo11x-pose.pt",
             "yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt", "yolov8l-pose.pt", "yolov8x-pose.pt"],
    "obb": ["yolo11n-obb.pt", "yolo11s-obb.pt", "yolo11m-obb.pt", "yolo11l-obb.pt", "yolo11x-obb.pt",
            "yolov8n-obb.pt", "yolov8s-obb.pt", "yolov8m-obb.pt", "yolov8l-obb.pt", "yolov8x-obb.pt"],
    "classify": ["yolo11n-cls.pt", "yolo11s-cls.pt", "yolo11m-cls.pt", "yolo11l-cls.pt", "yolo11x-cls.pt",
                 "yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt", "yolov8l-cls.pt", "yolov8x-cls.pt"],
}


def _downsample(points: List[tuple], max_points: int) -> List[tuple]:
    if len(points) <= max_points:
        return points
    step = len(points) / max_points
    return [points[int(i * step)] for i in range(max_points)]


def results_to_annotations(res, task: str, class_map: Optional[dict] = None,
                           kpt_conf: float = 0.5, max_points: int = 500) -> List[dict]:
    """ultralytics Results → 内部标注列表。class_map: {模型类别id: 数据集类别id}"""
    anns = []
    if task == "detect":
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            return anns
        for cls, xywhn, conf in zip(boxes.cls.tolist(), boxes.xywhn.tolist(), boxes.conf.tolist()):
            cid = int(cls)
            if class_map:
                cid = class_map.get(cid)
                if cid is None:
                    continue
            anns.append({"class_id": cid, "bbox": [round(v, 6) for v in xywhn], "confidence": round(conf, 4)})
    elif task == "segment":
        masks = getattr(res, "masks", None)
        boxes = getattr(res, "boxes", None)
        if masks is None or boxes is None:
            return anns
        for cls, xyn, conf in zip(boxes.cls.tolist(), masks.xyn.tolist(), boxes.conf.tolist()):
            cid = int(cls)
            if class_map:
                cid = class_map.get(cid)
                if cid is None:
                    continue
            pts = _downsample([(round(float(x), 6), round(float(y), 6)) for x, y in xyn], max_points)
            anns.append({"class_id": cid, "points": pts, "confidence": round(conf, 4)})
    elif task == "pose":
        boxes = getattr(res, "boxes", None)
        kpts = getattr(res, "keypoints", None)
        if boxes is None or kpts is None:
            return anns
        for cls, xywhn, conf, kp in zip(boxes.cls.tolist(), boxes.xywhn.tolist(), boxes.conf.tolist(), kpts.xyn.tolist()):
            cid = int(cls)
            if class_map:
                cid = class_map.get(cid)
                if cid is None:
                    continue
            keypoints = []
            for x, y, kconf in kp:
                v = 2 if kconf >= kpt_conf else 0
                keypoints.append([round(float(x), 6), round(float(y), 6), v])
            anns.append({"class_id": cid, "bbox": [round(v, 6) for v in xywhn], "keypoints": keypoints,
                         "confidence": round(conf, 4)})
    elif task == "obb":
        obb = getattr(res, "obb", None)
        if obb is None:
            return anns
        import math
        for cls, xywhr, conf in zip(obb.cls.tolist(), obb.xywhr.tolist(), obb.conf.tolist()):
            cid = int(cls)
            if class_map:
                cid = class_map.get(cid)
                if cid is None:
                    continue
            cx, cy, w, h, rad = xywhr
            angle = math.degrees(rad) % 180
            anns.append({"class_id": cid, "bbox": [round(float(cx), 6), round(float(cy), 6),
                                                   round(float(w), 6), round(float(h), 6)],
                         "angle": round(angle, 3), "confidence": round(conf, 4)})
    return anns


class AIAnnotateWorker(QThread):
    progress = Signal(int, int)      # (已完成, 总数)
    image_done = Signal(str, int)    # (图像名, 标注数)
    status = Signal(str)
    succeeded = Signal(int, int)     # (成功数, 总数)
    failed = Signal(str)

    def __init__(self, ds: Dataset, model_file: str, conf: float = 0.25, iou: float = 0.45,
                 max_det: int = 300, device: str = "auto", class_map: Optional[dict] = None,
                 kpt_conf: float = 0.5, max_points: int = 500, only_unannotated: bool = True,
                 pretrained_dir=None, parent=None):
        super().__init__(parent)
        self._ds = ds
        self._model_file = model_file
        self._pretrained_dir = pretrained_dir
        self._conf = conf
        self._iou = iou
        self._max_det = max_det
        self._device = device
        self._class_map = class_map
        self._kpt_conf = kpt_conf
        self._max_points = max_points
        self._only_unannotated = only_unannotated
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        try:
            from ultralytics import YOLO
            from services.model_utils import resolve_model_file
            model_path = resolve_model_file(self._model_file, self._pretrained_dir)
            if model_path != self._model_file:
                self.status.emit(f"使用本地模型 {model_path}")
            else:
                self.status.emit(f"正在下载/加载模型 {self._model_file} ...")
            model = YOLO(model_path)
            device = resolve_device(self._device)
            images = self._ds.image_files()
            if self._only_unannotated and self._ds.task != "classify":
                images = [i for i in images if not self._ds.read_annotations(i.name)]
            total = len(images)
            done = 0
            for img in images:
                if self._stop_flag:
                    break
                img_name = self._ds.image_display_name(img)
                results = model.predict(str(img), conf=self._conf, iou=self._iou, max_det=self._max_det,
                                        device=device, verbose=False)
                if self._ds.task == "classify":
                    if results and results[0].probs is not None:
                        top1 = int(results[0].probs.top1)
                        cid = self._class_map.get(top1, top1) if self._class_map else top1
                        cls_info = self._ds.class_by_id(cid)
                        if cls_info is not None:
                            self._ds.set_classify_label(img_name, cls_info.name)
                            self._ds.mark_ai(img_name)
                            self.image_done.emit(img_name, 1)
                else:
                    anns = results_to_annotations(results[0], self._ds.task, self._class_map,
                                                  self._kpt_conf, self._max_points)
                    if anns:
                        self._ds.write_annotations(img_name, anns)
                        self._ds.mark_ai(img_name)
                    self.image_done.emit(img_name, len(anns))
                done += 1
                self.progress.emit(done, total)
            self.status.emit("AI 标注完成")
            self.succeeded.emit(done, total)
        except Exception as e:
            logger.get_logger().exception("AI 标注失败")
            self.failed.emit(str(e))
