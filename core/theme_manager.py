# -*- coding: utf-8 -*-
"""主题管理：6 种预设主题，动态生成 QSS 样式表"""
from typing import List, Tuple

PALETTES = {
    "dark": {
        "name": "深色",
        "window": "#1e1f23", "panel": "#25272c", "sidebar": "#17181c", "sidebar_hover": "#2a2d33",
        "card": "#2b2d32", "border": "#3c3f46", "text": "#e8eaed", "text_dim": "#9aa0a8",
        "accent": "#4f8cff", "accent_hover": "#6ba1ff", "accent_text": "#ffffff",
        "input_bg": "#1a1b1f", "success": "#4caf7d", "warning": "#e6a23c", "error": "#e05c5c",
    },
    "light": {
        "name": "浅色",
        "window": "#f5f6f8", "panel": "#eceef1", "sidebar": "#e4e7eb", "sidebar_hover": "#d6dae0",
        "card": "#ffffff", "border": "#d5d9df", "text": "#22252a", "text_dim": "#6b7280",
        "accent": "#2f6fed", "accent_hover": "#1f5bd6", "accent_text": "#ffffff",
        "input_bg": "#ffffff", "success": "#2e9e6b", "warning": "#d97706", "error": "#dc2626",
    },
    "ocean": {
        "name": "海洋蓝",
        "window": "#0d2430", "panel": "#11303f", "sidebar": "#0a1c26", "sidebar_hover": "#17455a",
        "card": "#12384a", "border": "#1e5267", "text": "#e2f1f5", "text_dim": "#8fb3c0",
        "accent": "#14b8c6", "accent_hover": "#22d3e0", "accent_text": "#ffffff",
        "input_bg": "#0a2230", "success": "#34d399", "warning": "#fbbf24", "error": "#f87171",
    },
    "nature": {
        "name": "自然绿",
        "window": "#101d14", "panel": "#16281b", "sidebar": "#0c160f", "sidebar_hover": "#1e3a26",
        "card": "#1a2f1f", "border": "#2a4a33", "text": "#e8f3ea", "text_dim": "#93b39b",
        "accent": "#3da35d", "accent_hover": "#4dbb70", "accent_text": "#ffffff",
        "input_bg": "#0e1a12", "success": "#3da35d", "warning": "#fbbf24", "error": "#f87171",
    },
    "orange": {
        "name": "活力橙",
        "window": "#241a10", "panel": "#2f2316", "sidebar": "#1c140c", "sidebar_hover": "#3d2d1a",
        "card": "#332617", "border": "#4d3a22", "text": "#f7eee2", "text_dim": "#b09c82",
        "accent": "#e8862d", "accent_hover": "#f29a4b", "accent_text": "#ffffff",
        "input_bg": "#1f150d", "success": "#4caf7d", "warning": "#fbbf24", "error": "#f87171",
    },
    "purple": {
        "name": "优雅紫",
        "window": "#1c1726", "panel": "#251e33", "sidebar": "#151021", "sidebar_hover": "#2e2542",
        "card": "#291f3a", "border": "#3f3159", "text": "#efeaf7", "text_dim": "#a396c2",
        "accent": "#8b5cf6", "accent_hover": "#a07bfa", "accent_text": "#ffffff",
        "input_bg": "#171125", "success": "#34d399", "warning": "#fbbf24", "error": "#f87171",
    },
}

_QSS_TEMPLATE = """
* {{ outline: none; }}
QMainWindow, QDialog {{ background-color: {window}; color: {text}; }}
QWidget {{ color: {text}; font-family: "{font}"; font-size: {font_size}pt; }}
QToolTip {{ background-color: {card}; color: {text}; border: 1px solid {border}; padding: 4px 8px; }}
QFrame#sidebar {{ background-color: {sidebar}; border-right: 1px solid {border}; }}
QFrame#titlebar {{ background-color: {sidebar}; }}
QFrame#card {{ background-color: {card}; border: 1px solid {border}; border-radius: 8px; }}
QLabel#title {{ font-size: 16pt; font-weight: bold; color: {text}; }}
QLabel#subtitle {{ color: {text_dim}; }}
QLabel#dim {{ color: {text_dim}; }}
QLabel#success {{ color: {success}; }}
QLabel#warning {{ color: {warning}; }}
QLabel#error {{ color: {error}; }}
QPushButton {{ background-color: {card}; color: {text}; border: 1px solid {border}; border-radius: 6px; padding: 6px 14px; }}
QPushButton:hover {{ background-color: {accent}; color: {accent_text}; border-color: {accent}; }}
QPushButton:pressed {{ background-color: {accent_hover}; }}
QPushButton:disabled {{ color: {text_dim}; background-color: {panel}; border-color: {border}; }}
QPushButton#primary {{ background-color: {accent}; color: {accent_text}; border: none; font-weight: bold; }}
QPushButton#primary:hover {{ background-color: {accent_hover}; }}
QPushButton#primary:disabled {{ background-color: {border}; color: {text_dim}; }}
QPushButton#danger {{ background-color: transparent; color: {error}; border: 1px solid {error}; }}
QPushButton#danger:hover {{ background-color: {error}; color: #ffffff; }}
QPushButton#sidebarBtn {{ background-color: transparent; border: none; border-radius: 6px; text-align: left; padding: 10px 14px; color: {text_dim}; font-size: 11pt; }}
QPushButton#sidebarBtn:hover {{ background-color: {sidebar_hover}; color: {text}; }}
QPushButton#sidebarBtn:checked {{ background-color: {accent}; color: {accent_text}; }}
QPushButton#flat {{ background-color: transparent; border: none; }}
QPushButton#flat:hover {{ background-color: {sidebar_hover}; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{ background-color: {input_bg}; color: {text}; border: 1px solid {border}; border-radius: 5px; padding: 5px 8px; selection-background-color: {accent}; selection-color: {accent_text}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{ background-color: {card}; color: {text}; border: 1px solid {border}; selection-background-color: {accent}; selection-color: {accent_text}; }}
QTabWidget::pane {{ border: 1px solid {border}; border-radius: 6px; top: -1px; }}
QTabBar::tab {{ background: {panel}; color: {text_dim}; padding: 8px 18px; border: 1px solid {border}; border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px; }}
QTabBar::tab:selected {{ background: {card}; color: {accent}; font-weight: bold; border-top: 2px solid {accent}; }}
QTabBar::tab:hover {{ color: {text}; }}
QTableWidget, QTreeWidget, QListWidget, QListView {{ background-color: {card}; alternate-background-color: {panel}; gridline-color: {border}; color: {text}; border: 1px solid {border}; border-radius: 6px; }}
QHeaderView::section {{ background-color: {panel}; color: {text_dim}; border: none; border-bottom: 1px solid {border}; padding: 6px 10px; font-weight: bold; }}
QListWidget::item, QTreeWidget::item {{ padding: 6px 8px; border-radius: 4px; }}
QListWidget::item:selected, QTreeWidget::item:selected {{ background-color: {accent}; color: {accent_text}; }}
QListWidget::item:hover, QTreeWidget::item:hover {{ background-color: {sidebar_hover}; }}
QGroupBox {{ border: 1px solid {border}; border-radius: 8px; margin-top: 12px; padding-top: 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; color: {text_dim}; }}
QScrollBar:vertical {{ background: {panel}; width: 10px; border-radius: 5px; }}
QScrollBar::handle:vertical {{ background: {border}; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {text_dim}; }}
QScrollBar:horizontal {{ background: {panel}; height: 10px; border-radius: 5px; }}
QScrollBar::handle:horizontal {{ background: {border}; border-radius: 5px; min-width: 30px; }}
QScrollBar::handle:horizontal:hover {{ background: {text_dim}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QMenu {{ background-color: {card}; color: {text}; border: 1px solid {border}; border-radius: 6px; padding: 4px; }}
QMenu::item {{ padding: 6px 24px; border-radius: 4px; }}
QMenu::item:selected {{ background-color: {accent}; color: {accent_text}; }}
QProgressBar {{ background-color: {panel}; border: 1px solid {border}; border-radius: 6px; text-align: center; color: {text}; height: 18px; }}
QProgressBar::chunk {{ background-color: {accent}; border-radius: 5px; }}
QSlider::groove:horizontal {{ height: 6px; background: {panel}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {accent}; width: 14px; margin: -5px 0; border-radius: 7px; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 16px; height: 16px; border: 1px solid {border}; border-radius: 4px; background: {input_bg}; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{ background-color: {accent}; border-color: {accent}; }}
QGraphicsView {{ background-color: {input_bg}; border: 1px solid {border}; border-radius: 6px; }}
QStatusBar {{ background-color: {sidebar}; color: {text_dim}; }}
QSplitter::handle {{ background-color: {border}; }}
QAbstractScrollArea::corner {{ background: {panel}; }}
"""


class ThemeManager:
    @classmethod
    def list_themes(cls) -> List[Tuple[str, str]]:
        return [(k, v["name"]) for k, v in PALETTES.items()]

    @classmethod
    def exists(cls, key: str) -> bool:
        return key in PALETTES

    @classmethod
    def palette(cls, key: str) -> dict:
        if not cls.exists(key):
            key = "dark"
        return PALETTES[key]

    @classmethod
    def qss(cls, key: str, font: str = "Microsoft YaHei", font_size: int = 10) -> str:
        pal = cls.palette(key)
        pal = dict(pal)
        pal["font"] = font
        pal["font_size"] = font_size
        return _QSS_TEMPLATE.format(**pal)

    @classmethod
    def apply(cls, app, key: str, font: str = "Microsoft YaHei", font_size: int = 10):
        app.setStyleSheet(cls.qss(key, font, font_size))
