# -*- coding: utf-8 -*-
"""YOLO Platform V2.0 - 一体化人工智能视觉平台 · 主程序入口

启动：python main.py
"""
import os
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from core import logger as logger_mod
    from core.config import ConfigManager

    cm = ConfigManager(project_root)
    cm.load()
    log = logger_mod.setup_logging(cm.logs_root, cm.config.app.log_level)

    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        log.warning(f"当前 Python {sys.version.split()[0]} 不在推荐范围 3.10-3.12，可能出现兼容性问题")

    log.info(f"{cm.config.app.name} v{cm.config.app.version} 正在启动 ...")

    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setApplicationName(cm.config.app.name)
    app.setApplicationVersion(cm.config.app.version)

    from core.theme_manager import ThemeManager
    ThemeManager.apply(app, cm.config.ui.theme, cm.config.ui.font_family, cm.config.ui.font_size)

    # 登录
    from auth.manager import AuthManager
    auth = AuthManager(project_root / "workspace", cm.config.security.session_timeout_minutes)
    if auth.ensure_default_admin():
        log.info("首次运行，已创建默认管理员账号 admin / admin123（请尽快在设置中修改密码）")
    from gui.dialogs.login_dialog import LoginDialog
    login = LoginDialog(auth)
    if login.exec() != LoginDialog.Accepted:
        log.info("登录已取消，程序退出")
        sys.exit(0)

    # 主窗口
    from gui.main_window_modern import MainWindow
    win = MainWindow(cm, auth)
    win.resize(cm.config.ui.window_width, cm.config.ui.window_height)
    win.show()

    log.info("主窗口已启动")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
