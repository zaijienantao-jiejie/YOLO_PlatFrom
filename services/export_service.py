# -*- coding: utf-8 -*-
"""模型导出服务：ONNX / OpenVINO / TFLite / TensorRT / TorchScript / CoreML / Paddle"""
from PySide6.QtCore import QThread, Signal

from core import logger
from core.dependencies import resolve_device

EXPORT_FORMATS = {
    "onnx": "ONNX（跨平台）",
    "openvino": "OpenVINO（Intel 设备）",
    "tflite": "TensorFlow Lite（移动端/嵌入式）",
    "engine": "TensorRT（NVIDIA GPU）",
    "torchscript": "TorchScript",
    "coreml": "CoreML（Apple 设备）",
    "paddle": "PaddlePaddle",
}


class ExportWorker(QThread):
    status = Signal(str)
    succeeded = Signal(str)   # 导出文件路径
    failed = Signal(str)

    def __init__(self, weights: str, fmt: str, imgsz: int = 640, half: bool = False,
                 int8: bool = False, simplify: bool = True, dynamic: bool = False,
                 device: str = "auto", parent=None):
        super().__init__(parent)
        self._weights = weights
        self._fmt = fmt
        self._imgsz = imgsz
        self._half = half
        self._int8 = int8
        self._simplify = simplify
        self._dynamic = dynamic
        self._device = device

    def run(self):
        try:
            from ultralytics import YOLO
            self.status.emit(f"正在导出 {self._fmt} 格式 ...")
            model = YOLO(self._weights)
            kwargs = {
                "format": self._fmt,
                "imgsz": self._imgsz,
                "device": resolve_device(self._device),
            }
            if self._fmt == "onnx":
                kwargs.update({"half": self._half, "simplify": self._simplify, "dynamic": self._dynamic})
            elif self._fmt in ("engine", "tflite"):
                kwargs.update({"half": self._half, "int8": self._int8})
            elif self._fmt in ("openvino",):
                kwargs.update({"half": self._half, "int8": self._int8})
            exported = model.export(**kwargs)
            self.status.emit("导出完成")
            self.succeeded.emit(str(exported))
        except Exception as e:
            logger.get_logger().exception("导出失败")
            msg = str(e)
            if "tensorrt" in msg.lower() or "engine" in msg.lower():
                msg += "\n\n提示：TensorRT 导出需要 NVIDIA GPU 环境（可在 AutoDL 云 GPU 上执行）。"
            self.failed.emit(msg)
