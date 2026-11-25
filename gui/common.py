# -*- coding: utf-8 -*-
"""GUI 公共工具：任务信息、图像转换、布局与消息框"""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QMessageBox, QVBoxLayout, QWidget)

TASK_LABELS = {
    "classify": "图像分类",
    "detect": "目标检测",
    "segment": "实例分割",
    "pose": "姿态估计",
    "obb": "定向边界框",
}
TASK_ICONS = {"classify": "🖼", "detect": "📦", "segment": "🧩", "pose": "🕺", "obb": "🔄"}


def task_label(task: str) -> str:
    return TASK_LABELS.get(task, task)


def task_icon(task: str) -> str:
    return TASK_ICONS.get(task, "📁")


def np_to_pixmap(bgr) -> QPixmap:
    """BGR ndarray → QPixmap"""
    import cv2
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def load_pixmap(path, max_size=(1600, 900)) -> QPixmap:
    pm = QPixmap(str(path))
    if pm.isNull():
        return pm
    if pm.width() > max_size[0] or pm.height() > max_size[1]:
        pm = pm.scaled(max_size[0], max_size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pm


def color_hex(c) -> str:
    if isinstance(c, QColor):
        return c.name()
    return str(c)


def msg_info(parent, title: str, text: str):
    QMessageBox.information(parent, title, text)


def msg_warning(parent, title: str, text: str):
    QMessageBox.warning(parent, title, text)


def msg_error(parent, title: str, text: str):
    QMessageBox.critical(parent, title, text)


def msg_confirm(parent, title: str, text: str) -> bool:
    return QMessageBox.question(parent, title, text,
                                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes


def form_row(label_text: str, widget: QWidget) -> QWidget:
    """表单行：标签 + 控件"""
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 2, 0, 2)
    label = QLabel(label_text)
    label.setMinimumWidth(110)
    label.setObjectName("dim")
    lay.addWidget(label)
    lay.addWidget(widget, 1)
    return w


def vbox(*items, margins=(10, 10, 10, 10), spacing: int = 8, stretch=None) -> QVBoxLayout:
    lay = QVBoxLayout()
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    for it in items:
        if it is None:
            continue
        if isinstance(it, int):
            lay.addStretch(it)
        else:
            lay.addWidget(it)
    if stretch is not None:
        lay.addStretch(stretch)
    return lay


def hbox(*items, margins=(0, 0, 0, 0), spacing: int = 8, stretch=None) -> QHBoxLayout:
    lay = QHBoxLayout()
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    for it in items:
        if it is None:
            continue
        if isinstance(it, int):
            lay.addStretch(it)
        else:
            lay.addWidget(it)
    if stretch is not None:
        lay.addStretch(stretch)
    return lay
