# -*- coding: utf-8 -*-
"""日志系统：文件（轮转）+ 控制台 + 可选 Qt 桥接（GUI 实时显示）"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

APP_LOGGER = "app"
ACTIVITY_LOGGER = "activity"

_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")


def setup_logging(log_dir: Path, level: str = "INFO") -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(APP_LOGGER)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False

    fh = RotatingFileHandler(log_dir / "app.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    fh.setFormatter(_fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(_fmt)
    logger.addHandler(ch)

    alog = logging.getLogger(ACTIVITY_LOGGER)
    alog.setLevel(logging.INFO)
    alog.propagate = False
    if not alog.handlers:
        afh = RotatingFileHandler(log_dir / "activity.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
        afh.setFormatter(_fmt)
        alog.addHandler(afh)
    return logger


def get_logger(name: str = APP_LOGGER) -> logging.Logger:
    return logging.getLogger(name)


def attach_qt_handler():
    """将应用日志转发到 Qt 信号（返回带 entry 信号的 handler）"""
    from PySide6.QtCore import QObject, Signal

    class _LogBridge(QObject):
        entry = Signal(str, int)  # (文本, 级别)

    class QtLogHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.bridge = _LogBridge()
            self.entry = self.bridge.entry

        def emit(self, record):
            try:
                self.entry.emit(self.format(record), record.levelno)
            except RuntimeError:
                pass  # Qt 对象已被销毁

    logger = logging.getLogger(APP_LOGGER)
    handler = QtLogHandler()
    handler.setFormatter(_fmt)
    logger.addHandler(handler)
    return handler
