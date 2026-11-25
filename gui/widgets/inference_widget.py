# -*- coding: utf-8 -*-
"""推理部署面板：图像推理 / 视频与摄像头 / API 服务 / 模型导出"""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit,
                               QFileDialog, QTabWidget, QTextEdit)

from core import logger
from core.dependencies import is_full_install
from gui.common import form_row, msg_error, msg_info, np_to_pixmap
from gui.widgets.export_widget import ExportWidget
from services.inference_service import (ImagePredictWorker, VideoPredictWorker,
                                        APIServiceController)

TASK_LABELS = {"classify": "图像分类", "detect": "目标检测", "segment": "实例分割",
               "pose": "姿态估计", "obb": "定向边界框"}


class InferenceWidget(QWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.api_controller = APIServiceController()
        self.video_worker = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        title = QLabel("🌐 推理与部署")
        title.setObjectName("title")
        root.addWidget(title)
        tabs = QTabWidget()
        tabs.addTab(self._build_image_tab(), "🖼 图像推理")
        tabs.addTab(self._build_video_tab(), "🎥 视频 / 摄像头")
        tabs.addTab(self._build_api_tab(), "🔌 API 服务")
        tabs.addTab(ExportWidget(self.ctx), "📦 模型导出")
        root.addWidget(tabs, 1)

    # ---------------- 图像推理 ----------------
    def _build_image_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        bar = QHBoxLayout()
        self.img_model_edit = QLineEdit()
        self.img_model_edit.setPlaceholderText("选择 .pt 模型文件")
        btn_model = QPushButton("浏览...")
        btn_model.clicked.connect(lambda: self._pick_model(self.img_model_edit))
        self.img_path_edit = QLineEdit()
        self.img_path_edit.setPlaceholderText("选择图像文件")
        btn_img = QPushButton("浏览...")
        btn_img.clicked.connect(self._pick_image)
        bar.addWidget(self.img_model_edit, 2)
        bar.addWidget(btn_model)
        bar.addWidget(self.img_path_edit, 2)
        bar.addWidget(btn_img)
        lay.addLayout(bar)
        bar2 = QHBoxLayout()
        self.img_conf_spin = QDoubleSpinBox()
        self.img_conf_spin.setRange(0.01, 0.95)
        self.img_conf_spin.setValue(0.25)
        self.img_iou_spin = QDoubleSpinBox()
        self.img_iou_spin.setRange(0.01, 0.95)
        self.img_iou_spin.setValue(0.45)
        self.img_device_combo = QComboBox()
        self.img_device_combo.addItem("自动", "auto")
        self.img_device_combo.addItem("CPU", "cpu")
        self.img_device_combo.addItem("GPU 0", "0")
        self.btn_predict = QPushButton("▶ 开始推理")
        self.btn_predict.setObjectName("primary")
        self.btn_predict.clicked.connect(self.run_image_predict)
        bar2.addWidget(form_row("置信度", self.img_conf_spin))
        bar2.addWidget(form_row("IoU", self.img_iou_spin))
        bar2.addWidget(form_row("设备", self.img_device_combo))
        bar2.addWidget(self.btn_predict)
        bar2.addStretch(1)
        lay.addLayout(bar2)
        self.img_result_label = QLabel("推理结果将显示在这里")
        self.img_result_label.setAlignment(Qt.AlignCenter)
        self.img_result_label.setMinimumHeight(400)
        self.img_result_label.setObjectName("dim")
        lay.addWidget(self.img_result_label, 1)
        return w

    def _pick_model(self, edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", str(self.ctx.cm.models_root),
                                              "模型文件 (*.pt)")
        if path:
            edit.setText(path)

    def _pick_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "图像 (*.jpg *.jpeg *.png *.bmp *.webp)")
        if path:
            self.img_path_edit.setText(path)

    def run_image_predict(self):
        if not is_full_install():
            msg_error(self, "缺少依赖", "需要完整依赖（ultralytics/torch）才能推理")
            return
        model_path = self.img_model_edit.text().strip()
        img_path = self.img_path_edit.text().strip()
        if not model_path or not Path(model_path).exists():
            msg_error(self, "参数错误", "请选择模型文件")
            return
        if not img_path or not Path(img_path).exists():
            msg_error(self, "参数错误", "请选择图像文件")
            return
        self.btn_predict.setEnabled(False)
        self.worker = ImagePredictWorker(
            weights=model_path, image_path=img_path, conf=self.img_conf_spin.value(),
            iou=self.img_iou_spin.value(), device=self.img_device_combo.currentData(),
            out_dir=str(self.ctx.cm.outputs_root))
        self.worker.succeeded.connect(self._show_result_image)
        self.worker.failed.connect(lambda m: (self.btn_predict.setEnabled(True), msg_error(self, "推理失败", m)))
        self.worker.start()

    def _show_result_image(self, path: str):
        self.btn_predict.setEnabled(True)
        pm = QPixmap(path)
        self.img_result_label.setObjectName("")
        self.img_result_label.setPixmap(pm.scaled(self.img_result_label.size(),
                                                  Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # ---------------- 视频 / 摄像头 ----------------
    def _build_video_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        bar = QHBoxLayout()
        self.video_model_edit = QLineEdit()
        self.video_model_edit.setPlaceholderText("选择 .pt 模型文件")
        btn_model = QPushButton("浏览...")
        btn_model.clicked.connect(lambda: self._pick_model(self.video_model_edit))
        self.video_src_edit = QLineEdit()
        self.video_src_edit.setPlaceholderText("视频文件路径，或输入 0 使用默认摄像头")
        btn_src = QPushButton("浏览...")
        btn_src.clicked.connect(self._pick_video)
        bar.addWidget(self.video_model_edit, 2)
        bar.addWidget(btn_model)
        bar.addWidget(self.video_src_edit, 2)
        bar.addWidget(btn_src)
        lay.addLayout(bar)
        bar2 = QHBoxLayout()
        self.video_conf_spin = QDoubleSpinBox()
        self.video_conf_spin.setRange(0.01, 0.95)
        self.video_conf_spin.setValue(0.25)
        self.btn_video_start = QPushButton("▶ 开始")
        self.btn_video_start.setObjectName("primary")
        self.btn_video_start.clicked.connect(self.toggle_video)
        self.video_status = QLabel("就绪")
        self.video_status.setObjectName("dim")
        bar2.addWidget(form_row("置信度", self.video_conf_spin))
        bar2.addWidget(self.btn_video_start)
        bar2.addWidget(self.video_status, 1)
        lay.addLayout(bar2)
        self.video_label = QLabel("视频画面将显示在这里")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumHeight(360)
        self.video_label.setObjectName("dim")
        lay.addWidget(self.video_label, 1)
        return w

    def _pick_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频 (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self.video_src_edit.setText(path)

    def toggle_video(self):
        if self.video_worker is not None and self.video_worker.isRunning():
            self.video_worker.stop()
            return
        if not is_full_install():
            msg_error(self, "缺少依赖", "需要完整依赖（ultralytics/torch）才能推理")
            return
        model_path = self.video_model_edit.text().strip()
        if not model_path or not Path(model_path).exists():
            msg_error(self, "参数错误", "请选择模型文件")
            return
        src_text = self.video_src_edit.text().strip()
        source = int(src_text) if src_text.isdigit() else src_text
        if not src_text:
            msg_error(self, "参数错误", "请输入视频路径或摄像头索引（0）")
            return
        self.video_worker = VideoPredictWorker(weights=model_path, source=source,
                                               conf=self.video_conf_spin.value())
        self.video_worker.frame_ready.connect(self._show_video_frame)
        self.video_worker.stats.connect(lambda s: self.video_status.setText(f"FPS: {s['fps']}  帧数: {s['frames']}"))
        self.video_worker.failed.connect(lambda m: (self.video_status.setText("✖ " + m),))
        self.video_worker.finished.connect(lambda: (self.btn_video_start.setText("▶ 开始"),
                                                    self.video_status.setText("已结束")))
        self.video_worker.start()
        self.btn_video_start.setText("⏹ 停止")

    def _show_video_frame(self, bgr):
        pm = np_to_pixmap(bgr)
        self.video_label.setObjectName("")
        self.video_label.setPixmap(pm.scaled(self.video_label.size(), Qt.KeepAspectRatio,
                                             Qt.SmoothTransformation))

    # ---------------- API 服务 ----------------
    def _build_api_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        bar = QHBoxLayout()
        self.api_model_edit = QLineEdit()
        self.api_model_edit.setPlaceholderText("选择 .pt 模型文件")
        btn_model = QPushButton("浏览...")
        btn_model.clicked.connect(lambda: self._pick_model(self.api_model_edit))
        bar.addWidget(self.api_model_edit, 1)
        bar.addWidget(btn_model)
        lay.addLayout(bar)
        bar2 = QHBoxLayout()
        self.api_task_combo = QComboBox()
        for k, v in TASK_LABELS.items():
            self.api_task_combo.addItem(v, k)
        self.api_host_edit = QLineEdit(self.ctx.cm.config.security.api_host)
        self.api_port_spin = QSpinBox()
        self.api_port_spin.setRange(1024, 65535)
        self.api_port_spin.setValue(self.ctx.cm.config.security.api_port)
        self.btn_api = QPushButton("▶ 启动 API 服务")
        self.btn_api.setObjectName("primary")
        self.btn_api.clicked.connect(self.toggle_api)
        bar2.addWidget(form_row("任务类型", self.api_task_combo))
        bar2.addWidget(form_row("监听地址", self.api_host_edit))
        bar2.addWidget(form_row("端口", self.api_port_spin))
        bar2.addWidget(self.btn_api)
        lay.addLayout(bar2)
        self.api_status = QLabel("服务未启动")
        self.api_status.setObjectName("dim")
        lay.addWidget(self.api_status)
        info = QTextEdit()
        info.setReadOnly(True)
        info.setPlainText(
            "API 端点说明：\n"
            "  POST /predict        单图像推理（multipart 表单 file=@image.jpg）\n"
            "  POST /predict/batch  批量推理（files=@1.jpg files=@2.jpg）\n"
            "  POST /predict/stream 视频流推理（file=@video.mp4, frame_stride=N）\n"
            "  GET  /health         健康检查\n"
            "  GET  /info           模型信息\n"
            "  GET  /metrics        性能指标\n\n"
            "示例：curl -X POST http://127.0.0.1:8000/predict -F \"file=@image.jpg\"")
        lay.addWidget(info, 1)
        return w

    def toggle_api(self):
        if self.api_controller.is_running:
            self.api_controller.stop()
            self.btn_api.setText("▶ 启动 API 服务")
            self.api_status.setText("服务已停止")
            self.ctx.activity.log("停止 API 服务", "", "", actor=self.ctx.auth.current_username)
            return
        if not is_full_install():
            msg_error(self, "缺少依赖", "需要完整依赖（ultralytics/fastapi）才能启动 API 服务")
            return
        model_path = self.api_model_edit.text().strip()
        if not model_path or not Path(model_path).exists():
            msg_error(self, "参数错误", "请选择模型文件")
            return
        from deployment.api_server import create_app
        host = self.api_host_edit.text().strip() or "127.0.0.1"
        port = self.api_port_spin.value()
        try:
            app = create_app(model_path, task=self.api_task_combo.currentData(), device="auto")
            self.api_controller.start(app, host, port)
        except Exception as e:
            logger.get_logger().exception("API 服务启动失败")
            msg_error(self, "启动失败", str(e))
            return
        self.btn_api.setText("⏹ 停止 API 服务")
        self.api_status.setText(f"✔ 服务运行中：http://{host}:{port}  （文档: /docs）")
        self.ctx.activity.log("启动 API 服务", f"http://{host}:{port}", model_path,
                              actor=self.ctx.auth.current_username)

    def shutdown(self):
        """主窗口关闭时调用"""
        if self.video_worker is not None and self.video_worker.isRunning():
            self.video_worker.stop()
        if self.api_controller.is_running:
            self.api_controller.stop()
