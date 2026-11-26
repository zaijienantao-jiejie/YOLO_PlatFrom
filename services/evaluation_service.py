# -*- coding: utf-8 -*-
"""模型评估服务：验证集指标、混淆矩阵、PR 曲线、FPS 测试"""
import os
import time
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QThread, Signal

from core import logger
from core.dependencies import resolve_device


class EvaluationWorker(QThread):
    metrics = Signal(dict)          # 完整评估结果
    status = Signal(str)
    succeeded = Signal(str)         # 结果目录
    failed = Signal(str)

    def __init__(self, weights: str, data_yaml: str, task: str, imgsz: int = 640, batch: int = 16,
                 device: str = "auto", image_paths: Optional[List[Path]] = None, fps_samples: int = 50,
                 project: str = "workspace/outputs", parent=None):
        super().__init__(parent)
        self._weights = weights
        self._data_yaml = data_yaml
        self._task = task
        self._imgsz = imgsz
        self._batch = batch
        self._device = device
        self._image_paths = image_paths or []
        self._fps_samples = fps_samples
        self._project = project

    def run(self):
        try:
            from ultralytics import YOLO
            self.status.emit("正在加载模型 ...")
            model = YOLO(self._weights)
            run_name = f"val_{time.strftime('%Y%m%d_%H%M%S')}"
            self.status.emit("正在运行验证集评估 ...")
            results = model.val(data=self._data_yaml, split="val", imgsz=self._imgsz, batch=self._batch,
                                device=resolve_device(self._device), project=self._project, name=run_name,
                                exist_ok=True, plots=True, verbose=False)
            result_dict = {k: float(v) for k, v in (results.results_dict or {}).items()}

            # FPS 测试（在样本图像上跑前向）
            fps = None
            latency_ms = None
            samples = [str(p) for p in self._image_paths[:self._fps_samples]]
            if samples:
                self.status.emit("正在进行 FPS 性能测试 ...")
                t0 = time.time()
                model.predict(samples, imgsz=self._imgsz, device=resolve_device(self._device), verbose=False)
                dt = time.time() - t0
                fps = round(len(samples) / dt, 2)
                latency_ms = round(dt / len(samples) * 1000, 2)

            model_size_mb = round(os.path.getsize(self._weights) / 1024 / 1024, 2)
            save_dir = Path(self._project) / run_name
            self.metrics.emit({
                "results": result_dict,
                "fps": fps,
                "latency_ms": latency_ms,
                "model_size_mb": model_size_mb,
                "model_file": self._weights,
                "task": self._task,
                "results_dir": str(save_dir),
                "device": resolve_device(self._device),
            })
            self.status.emit("评估完成")
            self.succeeded.emit(str(save_dir))
        except Exception as e:
            logger.get_logger().exception("评估失败")
            self.failed.emit(str(e))
