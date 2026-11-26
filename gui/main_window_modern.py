# -*- coding: utf-8 -*-
"""现代化主窗口：无边框、MacBook 风格控制按钮、可折叠侧边栏、多工作区"""
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame,
                               QLabel, QPushButton, QStackedWidget, QTabWidget,
                               QButtonGroup, QSizeGrip, QToolTip)

from core import logger
from core.version import APP_NAME, APP_VERSION
from core.dependencies import torch_device_info
from gui.common import task_icon
from gui.context import AppContext
from gui.widgets.dataset_classify_workflow_widget import DatasetClassifyWorkflowWidget
from gui.widgets.dataset_detect_workflow_widget import DatasetDetectWorkflowWidget
from gui.widgets.dataset_segment_workflow_widget import DatasetSegmentWorkflowWidget
from gui.widgets.dataset_pose_workflow_widget import DatasetPoseWorkflowWidget
from gui.widgets.dataset_obb_workflow_widget import DatasetObbWorkflowWidget
from gui.widgets.training_widget import TrainingWidget
from gui.widgets.evaluation_widget import EvaluationWidget
from gui.widgets.inference_widget import InferenceWidget

PAGES = [
    ("datasets", "📁 数据集", "数据集"),
    ("training", "🚀 训练", "训练"),
    ("evaluation", "📊 评估", "评估"),
    ("deployment", "🌐 部署", "部署"),
    ("tools", "🛠 工具", "工具"),
]


class MainWindow(QMainWindow):
    def __init__(self, cm, auth):
        super().__init__()
        self.cm = cm
        self.auth = auth
        from core.activity_log import ActivityLogger
        from core.cache_manager import ThumbnailCache
        from data.dataset_manager import DatasetManager
        self.ctx = AppContext(
            cm=cm,
            auth=auth,
            dm=DatasetManager(cm.datasets_root),
            activity=ActivityLogger(cm.logs_root),
            cache=ThumbnailCache(cm.cache_root),
        )
        self.setWindowTitle(f"{APP_NAME} V{APP_VERSION}")
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self._build_ui()
        self._install_log_bridge()
        self._check_session_timer()
        logger.get_logger().info("主窗口初始化完成")

    # ---------------- UI ----------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 标题栏
        root.addWidget(self._build_titlebar())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        # 侧边栏
        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(210)
        sbar = QVBoxLayout(self.sidebar)
        sbar.setContentsMargins(8, 12, 8, 8)
        self.sidebar_header = QLabel(f"🎨 {APP_NAME.split(' ')[0]}")
        self.sidebar_header.setObjectName("title")
        sbar.addWidget(self.sidebar_header)
        ver_label = QLabel(f"V{APP_VERSION}")
        ver_label.setObjectName("dim")
        sbar.addWidget(ver_label)
        sbar.addSpacing(10)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons = {}
        for key, text, _ in PAGES:
            btn = QPushButton(text)
            btn.setObjectName("sidebarBtn")
            btn.setCheckable(True)
            btn.clicked.connect(lambda _, k=key: self.switch_page(k))
            self.nav_group.addButton(btn)
            self.nav_buttons[key] = btn
            sbar.addWidget(btn)
        sbar.addStretch(1)

        btn_collapse = QPushButton("◀ 收起侧边栏")
        btn_collapse.setObjectName("flat")
        btn_collapse.clicked.connect(self.toggle_sidebar)
        sbar.addWidget(btn_collapse)
        self.btn_collapse = btn_collapse
        body.addWidget(self.sidebar)

        # 页面堆栈
        self.stack = QStackedWidget()
        self.datasets_tabs = QTabWidget()
        self.datasets_tabs.addTab(DatasetClassifyWorkflowWidget(self.ctx), f"{task_icon('classify')} 图像分类")
        self.datasets_tabs.addTab(DatasetDetectWorkflowWidget(self.ctx), f"{task_icon('detect')} 目标检测")
        self.datasets_tabs.addTab(DatasetSegmentWorkflowWidget(self.ctx), f"{task_icon('segment')} 实例分割")
        self.datasets_tabs.addTab(DatasetPoseWorkflowWidget(self.ctx), f"{task_icon('pose')} 姿态估计")
        self.datasets_tabs.addTab(DatasetObbWorkflowWidget(self.ctx), f"{task_icon('obb')} 定向边界框")
        self.datasets_tabs.currentChanged.connect(self._on_dataset_tab_changed)
        self.stack.addWidget(self.datasets_tabs)
        self.training_widget = TrainingWidget(self.ctx)
        self.stack.addWidget(self.training_widget)
        self.evaluation_widget = EvaluationWidget(self.ctx)
        self.stack.addWidget(self.evaluation_widget)
        self.inference_widget = InferenceWidget(self.ctx)
        self.stack.addWidget(self.inference_widget)
        self.tools_widget = self._build_tools_page()
        self.stack.addWidget(self.tools_widget)
        body.addWidget(self.stack, 1)
        root.addLayout(body, 1)

        # 状态栏
        status = QFrame()
        status.setObjectName("sidebar")
        slay = QHBoxLayout(status)
        slay.setContentsMargins(12, 4, 12, 4)
        self.status_user = QLabel(f"👤 {self.auth.current_username}")
        self.status_user.setObjectName("dim")
        self.status_device = QLabel("")
        self.status_device.setObjectName("dim")
        self.status_msg = QLabel("就绪")
        self.status_msg.setObjectName("dim")
        slay.addWidget(self.status_user)
        slay.addWidget(self.status_device)
        slay.addStretch(1)
        slay.addWidget(self.status_msg)
        root.addWidget(status)

        self._update_device_status()
        self.switch_page("datasets")

    def _build_titlebar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("titlebar")
        bar.setFixedHeight(44)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(12, 0, 8, 0)
        self.title_label = QLabel(f"🎨 {APP_NAME} V{APP_VERSION}")
        self.title_label.setObjectName("title")
        lay.addWidget(self.title_label)
        lay.addStretch(1)
        # 控制按钮（MacBook 风格）
        self.btn_min = QPushButton("—")
        self.btn_min.setObjectName("flat")
        self.btn_min.setFixedSize(34, 30)
        self.btn_min.setToolTip("最小化")
        self.btn_min.clicked.connect(self.showMinimized)
        self.btn_max = QPushButton("□")
        self.btn_max.setObjectName("flat")
        self.btn_max.setFixedSize(34, 30)
        self.btn_max.setToolTip("最大化/还原")
        self.btn_max.clicked.connect(self._toggle_max)
        self.btn_close = QPushButton("✕")
        self.btn_close.setObjectName("flat")
        self.btn_close.setFixedSize(34, 30)
        self.btn_close.setToolTip("关闭")
        self.btn_close.clicked.connect(self.close)
        lay.addWidget(self.btn_min)
        lay.addWidget(self.btn_max)
        lay.addWidget(self.btn_close)
        # 拖动窗口
        bar.mousePressEvent = self._titlebar_press
        bar.mouseMoveEvent = self._titlebar_move
        bar.mouseDoubleClickEvent = lambda ev: self._toggle_max()
        return bar

    def _build_tools_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        title = QLabel("🛠 系统工具")
        title.setObjectName("title")
        lay.addWidget(title)
        cards = QHBoxLayout()
        from PySide6.QtWidgets import QFrame as QF
        for text, desc, fn in (
                ("🔧 系统诊断", "环境检查、硬件检测、依赖完整性", self._open_diagnostic),
                ("🧹 缓存清理", "清理字节码缓存、缩略图、临时文件", self._open_cache_cleanup),
                ("📜 日志查看器", "级别筛选、关键词搜索、日志导出", self._open_log_viewer),
                ("⚙️ 设置", "主题、训练默认值、API 配置、用户管理", self._open_settings),
        ):
            card = QF()
            card.setObjectName("card")
            cl = QVBoxLayout(card)
            btn = QPushButton(text)
            btn.setObjectName("flat")
            btn.setMinimumHeight(60)
            btn.clicked.connect(fn)
            cl.addWidget(btn)
            desc_label = QLabel(desc)
            desc_label.setObjectName("dim")
            desc_label.setWordWrap(True)
            cl.addWidget(desc_label)
            cards.addWidget(card)
        lay.addLayout(cards)
        lay.addStretch(1)
        return w

    # ---------------- 页面切换 ----------------
    def switch_page(self, key: str):
        index = {k: i for i, (k, _, _) in enumerate(PAGES)}[key]
        self.stack.setCurrentIndex(index)
        self.nav_buttons[key].setChecked(True)
        if key == "training":
            self.training_widget.refresh_datasets()
        elif key == "evaluation":
            self.evaluation_widget.refresh_datasets()

    def _on_dataset_tab_changed(self, idx):
        self.datasets_tabs.widget(idx).refresh()

    def toggle_sidebar(self):
        collapsed = self.sidebar.width() > 60
        self.sidebar.setFixedWidth(60 if collapsed else 210)
        self.btn_collapse.setText("▶" if collapsed else "◀ 收起侧边栏")
        for key, btn in self.nav_buttons.items():
            if collapsed:
                btn.setText(PAGES[[k for k, _, _ in PAGES].index(key)][1].split(" ")[0])
                self.sidebar_header.setText("🎨")
            else:
                btn.setText(PAGES[[k for k, _, _ in PAGES].index(key)][1])
                self.sidebar_header.setText(f"🎨 {APP_NAME.split(' ')[0]}")

    # ---------------- 工具对话框 ----------------
    def _open_diagnostic(self):
        from gui.dialogs.diagnostic_dialog import DiagnosticDialog
        DiagnosticDialog(self).exec()

    def _open_cache_cleanup(self):
        from gui.dialogs.cache_cleanup_dialog import CacheCleanupDialog
        CacheCleanupDialog(self.ctx, self).exec()

    def _open_log_viewer(self):
        from gui.dialogs.log_viewer_dialog import LogViewerDialog
        LogViewerDialog(self.ctx, self).exec()

    def _open_settings(self):
        from gui.dialogs.settings_dialog import SettingsDialog
        SettingsDialog(self.ctx, self).exec()

    # ---------------- 标题栏拖动 ----------------
    def _titlebar_press(self, ev):
        if ev.button() == Qt.LeftButton:
            self._drag_pos = ev.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def _titlebar_move(self, ev):
        if ev.buttons() & Qt.LeftButton and hasattr(self, "_drag_pos"):
            self.move(ev.globalPosition().toPoint() - self._drag_pos)

    def _toggle_max(self):
        if self.isMaximized():
            self.showNormal()
            self.btn_max.setText("□")
        else:
            self.showMaximized()
            self.btn_max.setText("❐")

    # ---------------- 状态 ----------------
    def _update_device_status(self):
        info = torch_device_info()
        if info["available"]:
            self.status_device.setText(f"🖥 {info['gpu']}")
        else:
            self.status_device.setText("🖥 CPU 模式")

    def _install_log_bridge(self):
        handler = logger.attach_qt_handler()
        handler.entry.connect(self._on_log_entry)

    def _on_log_entry(self, text: str, level: int):
        if level >= 40:  # ERROR 及以上显示在状态栏
            self.status_msg.setText(f"⚠ {text[-120:]}")

    def _check_session_timer(self):
        timer = QTimer(self)
        timer.timeout.connect(self._check_session)
        timer.start(60000)
        self._timer = timer

    def _check_session(self):
        if not self.auth.is_authenticated() and self.auth.current_user is not None:
            logger.get_logger().info("会话超时，已自动退出登录")
            self.status_user.setText("👤 已退出")
            self.auth.logout()

    # ---------------- 关闭 ----------------
    def closeEvent(self, ev):
        try:
            self.inference_widget.shutdown()
        except Exception:
            pass
        self.ctx.activity.log("退出应用", "", "", actor=self.auth.current_username)
        logger.get_logger().info("应用退出")
        super().closeEvent(ev)
