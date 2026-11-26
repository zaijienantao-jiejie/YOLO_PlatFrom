# -*- coding: utf-8 -*-
"""登录对话框"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QWidget)

from core import logger
from core.version import APP_NAME, APP_VERSION


class LoginDialog(QDialog):
    def __init__(self, auth, parent=None):
        super().__init__(parent)
        self.auth = auth
        self.setWindowTitle(f"登录 - {APP_NAME}")
        self.setFixedWidth(380)
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 24, 28, 24)
        lay.setSpacing(10)

        title = QLabel(f"🎨 {APP_NAME} V{APP_VERSION[:3]}")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)
        lay.addWidget(title)
        sub = QLabel("一体化人工智能视觉平台")
        sub.setObjectName("dim")
        sub.setAlignment(Qt.AlignCenter)
        lay.addWidget(sub)
        lay.addSpacing(10)

        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("用户名")
        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("密码")
        self.password_edit.setEchoMode(QLineEdit.Password)
        lay.addWidget(self.username_edit)
        lay.addWidget(self.password_edit)

        self.error_label = QLabel("")
        self.error_label.setObjectName("error")
        lay.addWidget(self.error_label)

        btn_login = QPushButton("登 录")
        btn_login.setObjectName("primary")
        btn_login.clicked.connect(self._login)
        lay.addWidget(btn_login)

        btn_guest = QPushButton("以访客身份使用")
        btn_guest.setObjectName("flat")
        btn_guest.clicked.connect(self._guest)
        lay.addWidget(btn_guest)

        hint = QLabel("默认管理员：admin / admin123（首次运行自动创建，登录后请在设置中修改）")
        hint.setObjectName("dim")
        hint.setWordWrap(True)
        hint.setAlignment(Qt.AlignCenter)
        lay.addWidget(hint)

        self.password_edit.returnPressed.connect(self._login)

    def _login(self):
        user, msg = self.auth.login(self.username_edit.text(), self.password_edit.text())
        if user is None:
            self.error_label.setText(msg)
            return
        logger.get_logger().info(f"用户 {user.username} 登录成功")
        self.accept()

    def _guest(self):
        self.auth.login_as_guest()
        logger.get_logger().info("以访客身份进入")
        self.accept()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._drag_pos = ev.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, ev):
        if ev.buttons() & Qt.LeftButton and hasattr(self, "_drag_pos"):
            self.move(ev.globalPosition().toPoint() - self._drag_pos)
