# -*- coding: utf-8 -*-
"""缓存清理对话框"""
import shutil
from pathlib import Path

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox)

from core import logger
from gui.common import msg_info


class CacheCleanupDialog(QDialog):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle("缓存清理")
        self.resize(520, 480)
        lay = QVBoxLayout(self)
        title = QLabel("🧹 缓存清理")
        title.setObjectName("title")
        lay.addWidget(title)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["清理项", "大小", "路径"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        lay.addWidget(self.table, 1)

        self.check_pycache = QCheckBox("Python 字节码缓存（__pycache__，会自动重建）")
        self.check_pycache.setChecked(True)
        self.check_thumb = QCheckBox("图像缩略图缓存")
        self.check_thumb.setChecked(True)
        self.check_logs = QCheckBox("旧日志文件")
        self.check_tmp = QCheckBox("系统临时文件（项目相关）")
        self.check_tmp.setChecked(True)
        for c in (self.check_pycache, self.check_thumb, self.check_logs, self.check_tmp):
            lay.addWidget(c)

        bar = QHBoxLayout()
        self.btn_scan = QPushButton("🔍 重新扫描")
        self.btn_scan.clicked.connect(self.scan)
        self.btn_clean = QPushButton("🧹 开始清理")
        self.btn_clean.setObjectName("primary")
        self.btn_clean.clicked.connect(self.clean)
        bar.addWidget(self.btn_scan)
        bar.addWidget(self.btn_clean)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.scan()

    def _dir_size(self, path: Path):
        if not path.exists():
            return 0
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    def scan(self):
        self.table.setRowCount(0)
        root = self.ctx.cm.project_root
        items = []
        pycache_dirs = list(root.rglob("__pycache__"))
        pycache_size = sum(self._dir_size(d) for d in pycache_dirs)
        items.append(("Python 字节码缓存", pycache_size, f"{len(pycache_dirs)} 个 __pycache__ 目录"))
        thumb_dir = self.ctx.cache.cache_dir
        thumb_size = self._dir_size(thumb_dir)
        items.append(("图像缩略图缓存", thumb_size, str(thumb_dir)))
        log_size = self._dir_size(root / "logs")
        items.append(("日志文件", log_size, str(root / "logs")))
        tmp = Path("workspace/cache") if (root / "workspace").exists() else None
        tmp_size = self._dir_size(root / "workspace" / "cache") if tmp else 0
        items.append(("工作区缓存", tmp_size, str(root / "workspace" / "cache")))
        for name, size, path in items:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem(f"{size / 1024 / 1024:.1f} MB"))
            self.table.setItem(row, 2, QTableWidgetItem(path))

    def clean(self):
        root = self.ctx.cm.project_root
        cleaned = 0
        if self.check_pycache.isChecked():
            for d in root.rglob("__pycache__"):
                shutil.rmtree(d, ignore_errors=True)
                cleaned += 1
        if self.check_thumb.isChecked():
            self.ctx.cache.clear()
        if self.check_logs.isChecked():
            log_dir = root / "logs"
            if log_dir.exists():
                for f in log_dir.glob("*.log.*"):
                    f.unlink(missing_ok=True)
        if self.check_tmp.isChecked():
            shutil.rmtree(root / "workspace" / "cache", ignore_errors=True)
        logger.get_logger().info(f"缓存清理完成（{cleaned} 个目录）")
        self.ctx.activity.log("缓存清理", "", f"{cleaned} 个目录", actor=self.ctx.auth.current_username)
        msg_info(self, "清理完成", "缓存已清理")
        self.scan()
