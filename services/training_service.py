# -*- coding: utf-8 -*-
"""模型训练服务：QThread 封装（GUI 使用）。核心逻辑见 services.train_core（无 GUI 依赖）"""
from PySide6.QtCore import QThread, Signal

from core import logger
from services.train_core import run_training, ensure_task_suffix  # noqa: F401（保持旧导入路径兼容）


class TrainingWorker(QThread):
    epoch = Signal(dict)          # 每个 epoch 的指标 {epoch, train/box_loss, mAP50, ...}
    status = Signal(str)          # 状态文本
    succeeded = Signal(str, str)  # (best 模型路径, 结果目录)
    failed = Signal(str)

    def __init__(self, data_yaml: str, model_name: str, params: dict, project: str,
                 device: str = "auto", resume: bool = False, pretrained_dir=None, parent=None):
        super().__init__(parent)
        self._data_yaml = data_yaml
        self._model_name = model_name
        self._pretrained_dir = pretrained_dir
        self._params = dict(params)
        self._project = project
        self._device = device
        self._resume = resume
        self._stop_flag = False
        self._stop_flag_obj = None

    def stop(self):
        self._stop_flag = True
        if self._stop_flag_obj is not None:
            self._stop_flag_obj["stop"] = True

    def run(self):
        try:
            self._stop_flag_obj = {"stop": False}
            best, save_dir = run_training(
                self._data_yaml, self._model_name, self._params, self._project,
                device=self._device, resume=self._resume, pretrained_dir=self._pretrained_dir,
                on_epoch=lambda out: self.epoch.emit(out),
                on_status=lambda msg: self.status.emit(msg),
                stop_flag=self._stop_flag_obj)
            if best is None:
                self.status.emit("训练已停止（可断点续训）")
                return
            self.succeeded.emit(best, save_dir)
        except Exception as e:
            logger.get_logger().exception("训练失败")
            self.failed.emit(str(e))
