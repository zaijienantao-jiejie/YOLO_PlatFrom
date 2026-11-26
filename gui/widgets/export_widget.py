# -*- coding: utf-8 -*-
"""模型导出面板：多格式导出（ONNX/OpenVINO/TFLite/TensorRT/TorchScript/CoreML/Paddle）"""
from pathlib import Path

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QComboBox, QSpinBox, QCheckBox, QLineEdit, QFileDialog)

from core.dependencies import is_full_install
from gui.common import form_row, msg_error, msg_info
from services.export_service import EXPORT_FORMATS, ExportWorker


class ExportWidget(QWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        title = QLabel("📦 模型导出")
        title.setObjectName("title")
        root.addWidget(title)

        bar = QHBoxLayout()
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("选择 .pt 模型文件")
        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self._browse)
        bar.addWidget(self.model_edit, 1)
        bar.addWidget(btn_browse)
        root.addLayout(bar)

        self.fmt_combo = QComboBox()
        for fmt, label in EXPORT_FORMATS.items():
            self.fmt_combo.addItem(label, fmt)
        self.fmt_combo.currentIndexChanged.connect(self._on_fmt_changed)
        root.addWidget(form_row("导出格式", self.fmt_combo))

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(64, 2048)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        root.addWidget(form_row("imgsz", self.imgsz_spin))

        opts = QHBoxLayout()
        self.half_check = QCheckBox("FP16 半精度")
        self.int8_check = QCheckBox("INT8 量化")
        self.simplify_check = QCheckBox("ONNX 简化")
        self.simplify_check.setChecked(True)
        self.dynamic_check = QCheckBox("动态输入尺寸")
        opts.addWidget(self.half_check)
        opts.addWidget(self.int8_check)
        opts.addWidget(self.simplify_check)
        opts.addWidget(self.dynamic_check)
        opts.addStretch(1)
        root.addLayout(opts)

        self.device_combo = QComboBox()
        self.device_combo.addItem("自动", "auto")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.addItem("GPU 0", "0")
        root.addWidget(form_row("设备", self.device_combo))

        self.hint_label = QLabel("")
        self.hint_label.setObjectName("dim")
        self.hint_label.setWordWrap(True)
        root.addWidget(self.hint_label)

        bar2 = QHBoxLayout()
        self.btn_run = QPushButton("▶ 开始导出")
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self.start_export)
        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("dim")
        bar2.addWidget(self.btn_run)
        bar2.addWidget(self.status_label, 1)
        root.addLayout(bar2)
        root.addStretch(1)
        self._on_fmt_changed()

    def _on_fmt_changed(self):
        fmt = self.fmt_combo.currentData()
        self.half_check.setEnabled(fmt in ("onnx", "openvino", "tflite", "engine"))
        self.int8_check.setEnabled(fmt in ("engine", "tflite", "openvino"))
        self.simplify_check.setEnabled(fmt == "onnx")
        self.dynamic_check.setEnabled(fmt == "onnx")
        hints = {
            "engine": "TensorRT 需要 NVIDIA GPU 环境（可在 AutoDL 云 GPU 上执行）。",
            "coreml": "CoreML 导出需要 macOS 环境。",
            "paddle": "Paddle 导出需要安装 paddlepaddle 依赖。",
        }
        self.hint_label.setText(hints.get(fmt, ""))

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", str(self.ctx.cm.models_root),
                                              "模型文件 (*.pt)")
        if path:
            self.model_edit.setText(path)

    def start_export(self):
        if not is_full_install():
            msg_error(self, "缺少依赖", "需要完整依赖（ultralytics/torch）才能导出模型")
            return
        model_path = self.model_edit.text().strip()
        if not model_path or not Path(model_path).exists():
            msg_error(self, "参数错误", "请选择有效的模型文件")
            return
        fmt = self.fmt_combo.currentData()
        self.btn_run.setEnabled(False)
        self.status_label.setText(f"正在导出 {fmt} ...")
        self.ctx.activity.log("导出模型", model_path, fmt, actor=self.ctx.auth.current_username)
        self.worker = ExportWorker(
            weights=model_path, fmt=fmt, imgsz=self.imgsz_spin.value(),
            half=self.half_check.isChecked() and self.half_check.isEnabled(),
            int8=self.int8_check.isChecked() and self.int8_check.isEnabled(),
            simplify=self.simplify_check.isChecked() and self.simplify_check.isEnabled(),
            dynamic=self.dynamic_check.isChecked() and self.dynamic_check.isEnabled(),
            device=self.device_combo.currentData())
        self.worker.status.connect(self.status_label.setText)
        self.worker.succeeded.connect(self._on_success)
        self.worker.failed.connect(self._on_fail)
        self.worker.start()

    def _on_success(self, path: str):
        self.btn_run.setEnabled(True)
        self.status_label.setText("✔ 导出完成")
        msg_info(self, "导出成功", f"模型已导出到：\n{path}")

    def _on_fail(self, msg: str):
        self.btn_run.setEnabled(True)
        self.status_label.setText("✖ 导出失败")
        msg_error(self, "导出失败", msg)
