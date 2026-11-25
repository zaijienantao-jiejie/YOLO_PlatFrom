# -*- coding: utf-8 -*-
"""模型训练面板：可视化配置 + 实时训练监控（损失/学习率/mAP 曲线）"""
import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
                               QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
                               QSplitter, QGroupBox, QFormLayout)

from core import logger
from core.dependencies import is_full_install, torch_device_info
from gui.common import form_row, msg_error, msg_info
from services.training_service import TrainingWorker, ensure_task_suffix

MODEL_SERIES = {
    "detect": [f"{s}{x}" for s in ("yolov8", "yolo11", "yolo12") for x in "nsmlx"],
    "segment": [f"{s}{x}-seg" for s in ("yolov8", "yolo11") for x in "nsmlx"],
    "pose": [f"{s}{x}-pose" for s in ("yolov8", "yolo11") for x in "nsmlx"],
    "obb": [f"{s}{x}-obb" for s in ("yolov8", "yolo11") for x in "nsmlx"],
    "classify": [f"{s}{x}-cls" for s in ("yolov8", "yolo11") for x in "nsmlx"],
}
DEFAULT_MODEL = {"detect": "yolo11n", "segment": "yolo11n-seg", "pose": "yolo11n-pose",
                 "obb": "yolo11n-obb", "classify": "yolo11n-cls"}


class TrainingWidget(QWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.worker = None
        self._history = []
        self._build_ui()
        self.refresh_datasets()

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Horizontal)

        # 左：配置面板
        panel = QWidget()
        panel.setMaximumWidth(420)
        lay = QVBoxLayout(panel)
        title = QLabel("🚀 模型训练")
        title.setObjectName("title")
        lay.addWidget(title)

        self.dataset_combo = QComboBox()
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_changed)
        lay.addWidget(form_row("数据集", self.dataset_combo))

        self.model_combo = QComboBox()
        lay.addWidget(form_row("模型架构", self.model_combo))

        grp = QGroupBox("训练参数")
        fl = QFormLayout(grp)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(self.ctx.cm.config.training.epochs)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(self.ctx.cm.config.training.batch_size)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(64, 2048)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(self.ctx.cm.config.training.image_size)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(4)
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setValue(self.ctx.cm.config.training.learning_rate)
        self.lrf_spin = QDoubleSpinBox()
        self.lrf_spin.setDecimals(3)
        self.lrf_spin.setRange(0.001, 0.1)
        self.lrf_spin.setValue(self.ctx.cm.config.training.final_lr_factor)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 500)
        self.patience_spin.setValue(self.ctx.cm.config.training.patience)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 32)
        self.workers_spin.setValue(min(self.ctx.cm.config.training.workers, 8))
        self.optimizer_combo = QComboBox()
        for o in ("auto", "SGD", "Adam", "AdamW"):
            self.optimizer_combo.addItem(o)
        self.device_combo = QComboBox()
        info = torch_device_info()
        if info["available"]:
            self.device_combo.addItem(f"自动（GPU: {info['gpu']}）", "auto")
        else:
            self.device_combo.addItem("自动（当前无 CUDA，将使用 CPU）", "auto")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.addItem("GPU 0", "0")
        self.amp_check = QCheckBox("混合精度训练（AMP，仅 GPU 生效）")
        self.amp_check.setChecked(self.ctx.cm.config.model.mixed_precision)
        self.resume_check = QCheckBox("断点续训（从 last.pt 恢复）")
        self.early_check = QCheckBox("早停机制")
        self.early_check.setChecked(self.ctx.cm.config.training.early_stopping)
        hint = QLabel("模型查找顺序：本地 → workspace/models/pretrained/ → ultralytics 官方目录；\n"
                      "都没有时首次训练会自动从 GitHub (ultralytics/assets) 下载预训练权重。")
        hint.setObjectName("dim")
        hint.setWordWrap(True)

        for label, w in (("epochs", self.epochs_spin), ("batch_size", self.batch_spin),
                         ("imgsz", self.imgsz_spin), ("lr0", self.lr_spin), ("lrf", self.lrf_spin),
                         ("patience", self.patience_spin), ("workers", self.workers_spin),
                         ("optimizer", self.optimizer_combo), ("device", self.device_combo)):
            fl.addRow(label, w)
        lay.addWidget(grp)
        lay.addWidget(self.amp_check)
        lay.addWidget(self.early_check)
        lay.addWidget(self.resume_check)
        lay.addWidget(hint)

        btns = QHBoxLayout()
        self.btn_start = QPushButton("▶ 开始训练")
        self.btn_start.setObjectName("primary")
        self.btn_start.clicked.connect(self.start_training)
        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_training)
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_stop)
        lay.addLayout(btns)

        self.status_label = QLabel("就绪")
        self.status_label.setWordWrap(True)
        lay.addWidget(self.status_label)
        self.progress = QProgressBar()
        lay.addWidget(self.progress)
        lay.addStretch(1)
        split.addWidget(panel)

        # 右：监控面板
        right = QWidget()
        rlay = QVBoxLayout(right)
        self.chart_canvas = self._make_chart_canvas()
        rlay.addWidget(self.chart_canvas, 3)
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["指标", "数值"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        rlay.addWidget(self.metrics_table, 2)
        split.addWidget(right)
        split.setSizes([400, 1000])
        root.addWidget(split)

    def _make_chart_canvas(self):
        import matplotlib
        matplotlib.use("QtAgg")
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from core.theme_manager import ThemeManager
        pal = ThemeManager.palette(self.ctx.cm.config.ui.theme)
        fig = Figure(figsize=(6, 4), dpi=100)
        fig.patch.set_facecolor(pal["window"])
        self.ax_loss = fig.add_subplot(121)
        self.ax_map = fig.add_subplot(122)
        for ax in (self.ax_loss, self.ax_map):
            ax.set_facecolor(pal["card"])
            ax.tick_params(colors=pal["text_dim"])
            for spine in ax.spines.values():
                spine.set_color(pal["border"])
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return FigureCanvasQTAgg(fig)

    # ---------------- 数据 ----------------
    def refresh_datasets(self):
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        for ds in self.ctx.dm.list():
            self.dataset_combo.addItem(f"{ds.name}（{ds.task}）", ds.name)
        self.dataset_combo.blockSignals(False)
        self._on_dataset_changed()

    def _on_dataset_changed(self):
        name = self.dataset_combo.currentData()
        if not name:
            return
        ds = self.ctx.dm.get(name)
        task = ds.task
        self.model_combo.clear()
        for m in MODEL_SERIES.get(task, MODEL_SERIES["detect"]):
            self.model_combo.addItem(m, m)
        default = DEFAULT_MODEL.get(task, "yolo11n")
        idx = self.model_combo.findData(default)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

    # ---------------- 训练 ----------------
    def start_training(self):
        if not is_full_install():
            msg_error(self, "缺少依赖", "需要完整依赖（ultralytics/torch）才能训练。请安装 requirements.txt")
            return
        name = self.dataset_combo.currentData()
        if not name:
            msg_error(self, "未选择", "请先选择数据集")
            return
        ds = self.ctx.dm.get(name)
        model_name = ensure_task_suffix(self.model_combo.currentData(), ds.task)
        if not ds.image_files():
            msg_error(self, "数据集为空", "请先向数据集添加图像")
            return
        try:
            if ds.task != "classify":
                counts = ds.generate_splits()
                if "train" not in counts:
                    msg_error(self, "划分失败", "训练集为空，请检查划分比例")
                    return
            yaml_path = ds.build_yaml()
        except Exception as e:
            msg_error(self, "数据准备失败", str(e))
            return

        params = {
            "epochs": self.epochs_spin.value(),
            "batch": self.batch_spin.value(),
            "imgsz": self.imgsz_spin.value(),
            "lr0": self.lr_spin.value(),
            "lrf": self.lrf_spin.value(),
            "optimizer": self.optimizer_combo.currentText(),
            "patience": self.patience_spin.value() if self.early_check.isChecked() else 0,
            "workers": self.workers_spin.value(),
            "amp": self.amp_check.isChecked(),
        }
        self._history = []
        self.metrics_table.setRowCount(0)
        self.ctx.activity.log("开始训练", ds.name, f"模型: {model_name}, epochs: {params['epochs']}",
                              actor=self.ctx.auth.current_username)
        self.worker = TrainingWorker(
            data_yaml=str(yaml_path), model_name=model_name, params=params,
            project=str(self.ctx.cm.models_root), device=self.device_combo.currentData(),
            resume=self.resume_check.isChecked(),
            pretrained_dir=str(self.ctx.cm.models_root / "pretrained"))
        self.worker.epoch.connect(self._on_epoch)
        self.worker.status.connect(self.status_label.setText)
        self.worker.succeeded.connect(self._on_success)
        self.worker.failed.connect(self._on_fail)
        self.worker.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setMaximum(params["epochs"])
        self.progress.setValue(0)

    def stop_training(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.status_label.setText("正在停止训练（当前 epoch 结束后生效）...")

    def _on_epoch(self, metrics: dict):
        self._history.append(metrics)
        ep = metrics.get("epoch", 0)
        self.progress.setValue(ep)
        # 更新指标表
        self.metrics_table.setRowCount(0)
        for k, v in metrics.items():
            row = self.metrics_table.rowCount()
            self.metrics_table.insertRow(row)
            self.metrics_table.setItem(row, 0, QTableWidgetItem(str(k)))
            if isinstance(v, float):
                self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{v:.4f}"))
            else:
                self.metrics_table.setItem(row, 1, QTableWidgetItem(str(v)))
        self._redraw_chart()

    def _redraw_chart(self):
        from core.theme_manager import ThemeManager
        pal = ThemeManager.palette(self.ctx.cm.config.ui.theme)
        self.ax_loss.clear()
        self.ax_map.clear()
        self.ax_loss.set_facecolor(pal["card"])
        self.ax_map.set_facecolor(pal["card"])
        for ax in (self.ax_loss, self.ax_map):
            ax.tick_params(colors=pal["text_dim"])
            for spine in ax.spines.values():
                spine.set_color(pal["border"])
            ax.grid(True, alpha=0.3)
        epochs = [h.get("epoch", i + 1) for i, h in enumerate(self._history)]
        loss_keys = [k for k in ("train/box_loss", "train/seg_loss", "train/cls_loss", "train/dfl_loss",
                                 "train/pose_loss", "train/obb_loss") if any(k in h for h in self._history)]
        colors = ["#4f8cff", "#e05c5c", "#4caf7d", "#e6a23c", "#8b5cf6", "#14b8c6"]
        for i, k in enumerate(loss_keys):
            vals = [h.get(k) for h in self._history]
            self.ax_loss.plot(epochs, vals, color=colors[i % len(colors)], label=k.split("/")[-1])
        if loss_keys:
            self.ax_loss.legend(fontsize=7, loc="upper right")
        self.ax_loss.set_title("损失曲线", color=pal["text"], fontsize=9)
        self.ax_loss.set_xlabel("epoch", color=pal["text_dim"], fontsize=8)
        map_keys = [k for k in ("mAP50", "mAP50-95", "top1", "precision", "recall", "mAP50-pose", "mAP50-95-pose")
                    if any(k in h for h in self._history)]
        for i, k in enumerate(map_keys):
            vals = [h.get(k) for h in self._history]
            self.ax_map.plot(epochs, vals, color=colors[(i + 2) % len(colors)], label=k)
        if map_keys:
            self.ax_map.legend(fontsize=7, loc="lower right")
        self.ax_map.set_title("精度指标", color=pal["text"], fontsize=9)
        self.ax_map.set_xlabel("epoch", color=pal["text_dim"], fontsize=8)
        self.chart_canvas.figure.tight_layout()
        self.chart_canvas.draw_idle()

    def _on_success(self, best_path: str, results_dir: str):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText("✔ 训练完成")
        self.ctx.activity.log("训练完成", "", best_path, actor=self.ctx.auth.current_username)
        msg_info(self, "训练完成", f"最佳模型已保存：\n{best_path}\n\n结果目录：\n{results_dir}\n\n可前往“评估”页评估模型，或在“部署”页导出/部署。")

    def _on_fail(self, msg: str):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText("✖ 训练失败")
        msg_error(self, "训练失败", msg)
