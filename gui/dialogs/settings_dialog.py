# -*- coding: utf-8 -*-
"""设置对话框：主题/界面、训练默认参数、API 配置、用户管理（管理员）"""
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QComboBox, QSpinBox, QLineEdit, QTabWidget, QWidget,
                               QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
                               QDialogButtonBox, QGroupBox, QFormLayout, QAbstractItemView)

from core.theme_manager import ThemeManager
from gui.common import form_row, msg_error, msg_info


class SettingsDialog(QDialog):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle("设置")
        self.resize(560, 620)
        lay = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self._build_ui_tab(), "🎨 界面")
        tabs.addTab(self._build_training_tab(), "🚀 训练默认值")
        tabs.addTab(self._build_api_tab(), "🔌 API 服务")
        if self.ctx.auth.current_user and self.ctx.auth.current_user.is_admin:
            tabs.addTab(self._build_users_tab(), "👥 用户管理")
        lay.addWidget(tabs)
        btns = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _build_ui_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        grp = QGroupBox("主题与字体")
        fl = QFormLayout(grp)
        self.theme_combo = QComboBox()
        current = self.ctx.cm.config.ui.theme
        for key, name in ThemeManager.list_themes():
            self.theme_combo.addItem(name, key)
        idx = self.theme_combo.findData(current)
        if idx >= 0:
            self.theme_combo.setCurrentIndex(idx)
        self.font_combo = QComboBox()
        for f in ("Microsoft YaHei", "SimHei", "Arial", "Segoe UI", "Noto Sans CJK SC"):
            self.font_combo.addItem(f)
        self.font_combo.setEditable(True)
        self.font_combo.setCurrentText(self.ctx.cm.config.ui.font_family)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 16)
        self.font_size_spin.setValue(self.ctx.cm.config.ui.font_size)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1000, 2560)
        self.width_spin.setValue(self.ctx.cm.config.ui.window_width)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(700, 1600)
        self.height_spin.setValue(self.ctx.cm.config.ui.window_height)
        fl.addRow("主题", self.theme_combo)
        fl.addRow("字体", self.font_combo)
        fl.addRow("字号", self.font_size_spin)
        fl.addRow("窗口宽度", self.width_spin)
        fl.addRow("窗口高度", self.height_spin)
        lay.addWidget(grp)
        grp2 = QGroupBox("会话")
        fl2 = QFormLayout(grp2)
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(0, 1440)
        self.timeout_spin.setSuffix(" 分钟（0 表示不超时）")
        self.timeout_spin.setValue(self.ctx.cm.config.security.session_timeout_minutes)
        fl2.addRow("会话超时", self.timeout_spin)
        lay.addWidget(grp2)
        lay.addStretch(1)
        return w

    def _build_training_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        grp = QGroupBox("训练默认参数（新建训练任务时的初始值）")
        fl = QFormLayout(grp)
        tc = self.ctx.cm.config.training
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(tc.epochs)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(tc.batch_size)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(64, 2048)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(tc.image_size)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 32)
        self.workers_spin.setValue(tc.workers)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 500)
        self.patience_spin.setValue(tc.patience)
        fl.addRow("epochs", self.epochs_spin)
        fl.addRow("batch_size", self.batch_spin)
        fl.addRow("imgsz", self.imgsz_spin)
        fl.addRow("workers", self.workers_spin)
        fl.addRow("patience", self.patience_spin)
        lay.addWidget(grp)
        lay.addStretch(1)
        return w

    def _build_api_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        grp = QGroupBox("API 服务默认配置")
        fl = QFormLayout(grp)
        self.api_host_edit = QLineEdit(self.ctx.cm.config.security.api_host)
        self.api_port_spin = QSpinBox()
        self.api_port_spin.setRange(1024, 65535)
        self.api_port_spin.setValue(self.ctx.cm.config.security.api_port)
        fl.addRow("监听地址", self.api_host_edit)
        fl.addRow("端口", self.api_port_spin)
        lay.addWidget(grp)
        hint = QLabel("注意：默认绑定 127.0.0.1（仅本机可访问），数据不出本机。")
        hint.setObjectName("dim")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        lay.addStretch(1)
        return w

    def _build_users_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.users_table = QTableWidget(0, 3)
        self.users_table.setHorizontalHeaderLabels(["用户名", "角色", "最后登录"])
        self.users_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.users_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.users_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._refresh_users()
        lay.addWidget(self.users_table, 1)
        bar = QHBoxLayout()
        btn_add = QPushButton("➕ 添加用户")
        btn_add.clicked.connect(self._add_user)
        btn_del = QPushButton("🗑 删除用户")
        btn_del.setObjectName("danger")
        btn_del.clicked.connect(self._delete_user)
        btn_pwd = QPushButton("🔑 修改密码")
        btn_pwd.clicked.connect(self._change_password)
        bar.addWidget(btn_add)
        bar.addWidget(btn_del)
        bar.addWidget(btn_pwd)
        bar.addStretch(1)
        lay.addLayout(bar)
        return w

    def _refresh_users(self):
        self.users_table.setRowCount(0)
        for u in self.ctx.auth.list_users():
            row = self.users_table.rowCount()
            self.users_table.insertRow(row)
            self.users_table.setItem(row, 0, QTableWidgetItem(u.username))
            self.users_table.setItem(row, 1, QTableWidgetItem(u.role))
            self.users_table.setItem(row, 2, QTableWidgetItem(u.last_login or "-"))

    def _selected_username(self):
        row = self.users_table.currentRow()
        if row < 0:
            msg_error(self, "未选择", "请先选择用户")
            return None
        return self.users_table.item(row, 0).text()

    def _add_user(self):
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "添加用户", "用户名：")
        if not ok or not name.strip():
            return
        pwd, ok2 = QInputDialog.getText(self, "添加用户", "密码（至少 6 位）：")
        if not ok2:
            return
        ok3, msg = self.ctx.auth.add_user(name.strip(), pwd)
        (msg_info if ok3 else msg_error)(self, "添加用户", msg)
        self._refresh_users()

    def _delete_user(self):
        name = self._selected_username()
        if not name:
            return
        if QMessageBox.question(self, "确认删除", f"确定删除用户 {name}？") != QMessageBox.Yes:
            return
        ok, msg = self.ctx.auth.delete_user(name)
        (msg_info if ok else msg_error)(self, "删除用户", msg)
        self._refresh_users()

    def _change_password(self):
        from PySide6.QtWidgets import QInputDialog
        name = self._selected_username()
        if not name:
            return
        old, ok = QInputDialog.getText(self, "修改密码", f"{name} 的原密码：")
        if not ok:
            return
        new, ok2 = QInputDialog.getText(self, "修改密码", f"{name} 的新密码（至少 6 位）：")
        if not ok2:
            return
        ok3, msg = self.ctx.auth.change_password(name, old, new)
        (msg_info if ok3 else msg_error)(self, "修改密码", msg)

    def _save(self):
        cm = self.ctx.cm
        cm.config.ui.theme = self.theme_combo.currentData()
        cm.config.ui.font_family = self.font_combo.currentText()
        cm.config.ui.font_size = self.font_size_spin.value()
        cm.config.ui.window_width = self.width_spin.value()
        cm.config.ui.window_height = self.height_spin.value()
        cm.config.security.session_timeout_minutes = self.timeout_spin.value()
        cm.config.security.api_host = self.api_host_edit.text().strip() or "127.0.0.1"
        cm.config.security.api_port = self.api_port_spin.value()
        cm.config.training.epochs = self.epochs_spin.value()
        cm.config.training.batch_size = self.batch_spin.value()
        cm.config.training.image_size = self.imgsz_spin.value()
        cm.config.training.workers = self.workers_spin.value()
        cm.config.training.patience = self.patience_spin.value()
        cm.save()
        # 立即应用主题
        from PySide6.QtWidgets import QApplication
        ThemeManager.apply(QApplication.instance(), cm.config.ui.theme,
                           cm.config.ui.font_family, cm.config.ui.font_size)
        self.ctx.activity.log("修改设置", "settings", "", actor=self.ctx.auth.current_username)
        msg_info(self, "已保存", "设置已保存并生效")
        self.accept()
