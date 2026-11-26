# -*- coding: utf-8 -*-
"""AI 辅助标注对话框：预训练模型批量预标注 + 类别映射"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
                               QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton,
                               QProgressBar, QTableWidget, QTableWidgetItem, QDialogButtonBox,
                               QFileDialog, QTextEdit, QHeaderView)

from core import logger
from core.dependencies import resolve_device
from gui.common import form_row, msg_error, msg_info
from services.annotation_service import PRETRAINED_MODELS, COCO_NAMES, AIAnnotateWorker


class AIAnnotateDialog(QDialog):
    def __init__(self, ctx, ds, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.ds = ds
        self.worker = None
        self.setWindowTitle(f"AI 辅助标注 - {ds.name}")
        self.resize(640, 600)
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        self.model_combo = QComboBox()
        lay.addWidget(form_row("预训练模型", self.model_combo))
        for m in PRETRAINED_MODELS.get(self.ds.task, []):
            self.model_combo.addItem(m, m)
        btn_row = QHBoxLayout()
        self.btn_custom = QPushButton("使用本地模型...")
        self.btn_custom.clicked.connect(self._pick_custom_model)
        btn_row.addWidget(self.btn_custom)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

        grid = QHBoxLayout()
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.95)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setSingleStep(0.05)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 0.95)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setSingleStep(0.05)
        self.max_det_spin = QSpinBox()
        self.max_det_spin.setRange(1, 1000)
        self.max_det_spin.setValue(300)
        grid.addWidget(form_row("置信度阈值", self.conf_spin))
        grid.addWidget(form_row("IoU 阈值", self.iou_spin))
        grid.addWidget(form_row("最大检测数", self.max_det_spin))
        lay.addLayout(grid)

        if self.ds.task == "pose":
            self.kpt_conf_spin = QDoubleSpinBox()
            self.kpt_conf_spin.setRange(0.1, 0.95)
            self.kpt_conf_spin.setValue(0.5)
            self.kpt_conf_spin.setSingleStep(0.05)
            lay.addWidget(form_row("关键点置信度阈值", self.kpt_conf_spin))
        if self.ds.task == "segment":
            self.max_points_spin = QSpinBox()
            self.max_points_spin.setRange(10, 2000)
            self.max_points_spin.setValue(500)
            lay.addWidget(form_row("最大轮廓点数", self.max_points_spin))

        self.device_combo = QComboBox()
        lay.addWidget(form_row("运行设备", self.device_combo))
        self.device_combo.addItem("自动（GPU 优先，否则 CPU）", "auto")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.addItem("GPU 0", "0")

        self.only_unlabeled_check = QCheckBox("仅标注未标注的图像")
        self.only_unlabeled_check.setChecked(True)
        lay.addWidget(self.only_unlabeled_check)

        hint = QLabel("模型查找顺序：本地文件 → workspace/models/pretrained/ → ultralytics 官方目录；\n"
                      "都没有时自动从 GitHub (ultralytics/assets) 下载预训练权重。")
        hint.setObjectName("dim")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        # 类别映射
        lay.addWidget(QLabel("类别映射（模型类别 → 数据集类别，留空表示直接按 COCO 类别对应）："))
        self.map_table = QTableWidget(len(self.ds.meta.classes), 2)
        self.map_table.setHorizontalHeaderLabels(["数据集类别", "模型类别"])
        self.map_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for i, c in enumerate(self.ds.meta.classes):
            self.map_table.setItem(i, 0, QTableWidgetItem(c.name))
            combo = QComboBox()
            combo.addItem("（忽略该类别）", None)
            for j, name in enumerate(COCO_NAMES):
                combo.addItem(f"{j}: {name}", j)
            # 自动匹配同名类别
            match = None
            for j, name in enumerate(COCO_NAMES):
                if name == c.name:
                    match = j
                    break
            if match is not None:
                combo.setCurrentIndex(match + 1)
            self.map_table.setCellWidget(i, 1, combo)
        lay.addWidget(self.map_table)

        self.progress = QProgressBar()
        lay.addWidget(self.progress)
        self.status_label = QLabel("")
        self.status_label.setObjectName("dim")
        lay.addWidget(self.status_label)

        btns = QHBoxLayout()
        self.btn_run = QPushButton("🚀 开始 AI 标注")
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self._run)
        self.btn_close = QPushButton("关闭")
        self.btn_close.clicked.connect(self.reject)
        btns.addWidget(self.btn_run)
        btns.addWidget(self.btn_close)
        lay.addLayout(btns)

    def _pick_custom_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "模型文件 (*.pt *.onnx)")
        if path:
            self.model_combo.addItem(f"自定义: {path}", path)
            self.model_combo.setCurrentIndex(self.model_combo.count() - 1)

    def _class_map(self):
        cmap = {}
        for i in range(self.map_table.rowCount()):
            combo = self.map_table.cellWidget(i, 1)
            if combo is None or combo.currentData() is None:
                continue
            cmap[combo.currentData()] = i
        return cmap or None

    def _run(self):
        from core.dependencies import is_full_install
        if not is_full_install():
            msg_error(self, "缺少依赖", "需要安装完整依赖（ultralytics/torch）才能使用 AI 标注")
            return
        model = self.model_combo.currentData()
        params = {
            "conf": self.conf_spin.value(), "iou": self.iou_spin.value(),
            "max_det": self.max_det_spin.value(), "device": self.device_combo.currentData(),
            "class_map": self._class_map(),
            "only_unannotated": self.only_unlabeled_check.isChecked(),
        }
        if self.ds.task == "pose":
            params["kpt_conf"] = self.kpt_conf_spin.value()
        if self.ds.task == "segment":
            params["max_points"] = self.max_points_spin.value()
        self.btn_run.setEnabled(False)
        self.worker = AIAnnotateWorker(self.ds, model, pretrained_dir=str(self.ctx.cm.models_root / "pretrained"), **params)
        self.worker.progress.connect(lambda done, total: (self.progress.setMaximum(max(total, 1)), self.progress.setValue(done)))
        self.worker.status.connect(self.status_label.setText)
        self.worker.succeeded.connect(self._on_done)
        self.worker.failed.connect(self._on_fail)
        self.worker.start()

    def _on_done(self, done, total):
        self.btn_run.setEnabled(True)
        self.ctx.activity.log("AI 标注", self.ds.name, f"{done}/{total} 张图像",
                              actor=self.ctx.auth.current_username)
        msg_info(self, "完成", f"AI 标注完成：共处理 {done}/{total} 张图像。\n请在标注界面中审核修正。")
        self.accept()

    def _on_fail(self, msg):
        self.btn_run.setEnabled(True)
        msg_error(self, "AI 标注失败", msg)

    def closeEvent(self, ev):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        super().closeEvent(ev)
