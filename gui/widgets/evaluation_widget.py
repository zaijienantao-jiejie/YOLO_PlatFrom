# -*- coding: utf-8 -*-
"""模型评估面板：验证集指标、混淆矩阵、PR 曲线、FPS 性能测试"""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QComboBox, QSpinBox, QTableWidget, QTableWidgetItem,
                               QHeaderView, QFileDialog, QSplitter, QListWidget,
                               QListWidgetItem)

from core.dependencies import is_full_install
from gui.common import form_row, msg_error, msg_info, load_pixmap
from services.evaluation_service import EvaluationWorker


class EvaluationWidget(QWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.worker = None
        self._last_metrics = None
        self._build_ui()
        self.refresh_datasets()

    def _build_ui(self):
        root = QVBoxLayout(self)
        title = QLabel("📊 模型评估")
        title.setObjectName("title")
        root.addWidget(title)

        bar = QHBoxLayout()
        from PySide6.QtWidgets import QLineEdit
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("选择 best.pt / last.pt 模型文件")
        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self._browse_model)
        self.dataset_combo = QComboBox()
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(64, 2048)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(16)
        self.device_combo = QComboBox()
        self.device_combo.addItem("自动", "auto")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.addItem("GPU 0", "0")
        bar.addWidget(self.model_edit, 2)
        bar.addWidget(btn_browse)
        bar.addWidget(form_row("数据集", self.dataset_combo))
        bar.addWidget(form_row("imgsz", self.imgsz_spin))
        bar.addWidget(form_row("batch", self.batch_spin))
        bar.addWidget(form_row("设备", self.device_combo))
        root.addLayout(bar)

        bar2 = QHBoxLayout()
        self.btn_run = QPushButton("▶ 开始评估")
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self.start_evaluation)
        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("dim")
        bar2.addWidget(self.btn_run)
        bar2.addWidget(self.status_label, 1)
        root.addLayout(bar2)

        split = QSplitter(Qt.Horizontal)
        # 指标表
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["指标", "数值"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        split.addWidget(self.metrics_table)
        # 图表区
        self.chart_list = QListWidget()
        self.chart_list.itemDoubleClicked.connect(self._open_chart)
        split.addWidget(self.chart_list)
        split.setSizes([500, 600])
        root.addWidget(split, 1)

    def refresh_datasets(self):
        self.dataset_combo.clear()
        for ds in self.ctx.dm.list():
            if ds.task == "classify":
                continue
            self.dataset_combo.addItem(f"{ds.name}（{ds.task}）", ds.name)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", str(self.ctx.cm.models_root),
                                              "模型文件 (*.pt)")
        if path:
            self.model_edit.setText(path)

    def start_evaluation(self):
        if not is_full_install():
            msg_error(self, "缺少依赖", "需要完整依赖（ultralytics/torch）才能评估模型")
            return
        model_path = self.model_edit.text().strip()
        if not model_path or not Path(model_path).exists():
            msg_error(self, "参数错误", "请选择有效的模型文件")
            return
        ds_name = self.dataset_combo.currentData()
        if not ds_name:
            msg_error(self, "参数错误", "请选择数据集")
            return
        ds = self.ctx.dm.get(ds_name)
        try:
            if ds.task != "classify":
                counts = ds.generate_splits()
                if "val" not in counts:
                    msg_error(self, "划分失败", "验证集为空，请调整划分比例后重试")
                    return
            yaml_path = ds.training_data_path()
        except Exception as e:
            msg_error(self, "数据准备失败", str(e))
            return

        self.btn_run.setEnabled(False)
        self.status_label.setText("评估中 ...")
        self.metrics_table.setRowCount(0)
        self.chart_list.clear()
        self.ctx.activity.log("开始评估", ds.name, model_path, actor=self.ctx.auth.current_username)
        self.worker = EvaluationWorker(
            weights=model_path, data_yaml=str(yaml_path), task=ds.task,
            imgsz=self.imgsz_spin.value(), batch=self.batch_spin.value(),
            device=self.device_combo.currentData(), image_paths=ds.image_files(),
            project=str(self.ctx.cm.outputs_root))
        self.worker.status.connect(self.status_label.setText)
        self.worker.metrics.connect(self._on_metrics)
        self.worker.failed.connect(self._on_fail)
        self.worker.start()

    def _on_metrics(self, metrics: dict):
        self._last_metrics = metrics
        self.metrics_table.setRowCount(0)
        # 附加信息
        for label, key in (("模型文件", "model_file"), ("运行设备", "device"),
                           ("FPS（推理速度）", "fps"), ("平均延迟 (ms)", "latency_ms"),
                           ("模型大小 (MB)", "model_size_mb")):
            val = metrics.get(key)
            if val is not None:
                self._add_row(label, str(val))
        for k, v in sorted(metrics.get("results", {}).items()):
            self._add_row(k, f"{v:.4f}")
        # 图表
        results_dir = Path(metrics.get("results_dir", ""))
        self.chart_list.clear()
        if results_dir.exists():
            for png in sorted(results_dir.glob("*.png")):
                item = QListWidgetItem(png.name)
                item.setToolTip(str(png))
                self.chart_list.addItem(item)
        self.status_label.setText("✔ 评估完成")

    def _add_row(self, k: str, v: str):
        row = self.metrics_table.rowCount()
        self.metrics_table.insertRow(row)
        self.metrics_table.setItem(row, 0, QTableWidgetItem(str(k)))
        self.metrics_table.setItem(row, 1, QTableWidgetItem(v))

    def _open_chart(self, item: QListWidgetItem):
        path = Path(item.toolTip())
        if not path.exists():
            return
        from PySide6.QtWidgets import QDialog, QLabel as QL
        dlg = QDialog(self)
        dlg.setWindowTitle(path.name)
        lay = QVBoxLayout(dlg)
        lab = QL()
        lab.setPixmap(load_pixmap(path, max_size=(1200, 1200)))
        lay.addWidget(lab)
        dlg.resize(1200, 900)
        dlg.exec()

    def _on_fail(self, msg: str):
        self.btn_run.setEnabled(True)
        self.status_label.setText("✖ 评估失败")
        msg_error(self, "评估失败", msg)
