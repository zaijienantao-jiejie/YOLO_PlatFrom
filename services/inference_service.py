# -*- coding: utf-8 -*-
"""本地推理服务：图像 / 视频 / 摄像头推理，以及 API 服务控制"""
import threading
import time
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from core import logger
from core.dependencies import resolve_device


class ImagePredictWorker(QThread):
    succeeded = Signal(str)   # 结果图像路径
    failed = Signal(str)

    def __init__(self, weights: str, image_path: str, conf: float = 0.25, iou: float = 0.45,
                 device: str = "auto", out_dir: str = "workspace/outputs", parent=None):
        super().__init__(parent)
        self._weights = weights
        self._image_path = image_path
        self._conf = conf
        self._iou = iou
        self._device = device
        self._out_dir = out_dir

    def run(self):
        try:
            import cv2
            from ultralytics import YOLO
            model = YOLO(self._weights)
            results = model.predict(self._image_path, conf=self._conf, iou=self._iou,
                                    device=resolve_device(self._device), verbose=False)
            out = Path(self._out_dir) / f"predict_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            annotated = results[0].plot()
            cv2.imwrite(str(out), annotated)
            self.succeeded.emit(str(out))
        except Exception as e:
            logger.get_logger().exception("推理失败")
            self.failed.emit(str(e))


class VideoPredictWorker(QThread):
    frame_ready = Signal(object)   # BGR ndarray
    stats = Signal(dict)           # {fps, frames}
    failed = Signal(str)
    finished = Signal()

    def __init__(self, weights: str, source, conf: float = 0.25, iou: float = 0.45,
                 device: str = "auto", parent=None):
        """source: 视频路径或摄像头索引（0 表示默认摄像头）"""
        super().__init__(parent)
        self._weights = weights
        self._source = source
        self._conf = conf
        self._iou = iou
        self._device = device
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        try:
            import cv2
            from ultralytics import YOLO
            model = YOLO(self._weights)
            cap = cv2.VideoCapture(self._source)
            if not cap.isOpened():
                self.failed.emit(f"无法打开视频源: {self._source}")
                return
            frames = 0
            t0 = time.time()
            while not self._stop_flag:
                ok, frame = cap.read()
                if not ok:
                    break
                results = model.predict(frame, conf=self._conf, iou=self._iou,
                                        device=resolve_device(self._device), verbose=False)
                annotated = results[0].plot()
                self.frame_ready.emit(annotated)
                frames += 1
                if frames % 30 == 0:
                    fps = frames / max(time.time() - t0, 1e-6)
                    self.stats.emit({"fps": round(fps, 2), "frames": frames})
            cap.release()
            self.stats.emit({"fps": round(frames / max(time.time() - t0, 1e-6), 2), "frames": frames})
            self.finished.emit()
        except Exception as e:
            logger.get_logger().exception("视频推理失败")
            self.failed.emit(str(e))


class APIServiceController:
    """在后台线程运行 FastAPI + Uvicorn 推理服务"""

    def __init__(self):
        self._thread: threading.Thread = None
        self._server = None

    def start(self, app, host: str, port: int):
        import uvicorn
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        logger.get_logger().info(f"API 服务已启动: http://{host}:{port}")

    @property
    def is_running(self) -> bool:
        return self._server is not None and self._server.started and not self._server.should_exit

    def stop(self):
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None
        logger.get_logger().info("API 服务已停止")
