# -*- coding: utf-8 -*-
"""日志查看器对话框：级别筛选、关键词搜索、导出"""
import csv
from pathlib import Path

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QComboBox, QLineEdit, QPlainTextEdit, QFileDialog)

from gui.common import msg_info

LEVELS = {"全部": None, "DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}


class LogViewerDialog(QDialog):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle("日志查看器")
        self.resize(860, 560)
        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.level_combo = QComboBox()
        for name in LEVELS:
            self.level_combo.addItem(name)
        self.keyword_edit = QLineEdit()
        self.keyword_edit.setPlaceholderText("关键词搜索")
        self.log_combo = QComboBox()
        self.log_combo.addItem("app.log（应用日志）", "app.log")
        self.log_combo.addItem("activity.log（操作日志）", "activity.log")
        bar.addWidget(QLabel("级别:"))
        bar.addWidget(self.level_combo)
        bar.addWidget(self.keyword_edit, 1)
        bar.addWidget(self.log_combo)
        btn_refresh = QPushButton("🔄 刷新")
        btn_refresh.clicked.connect(self.load_logs)
        btn_export = QPushButton("📤 导出")
        btn_export.clicked.connect(self.export_logs)
        bar.addWidget(btn_refresh)
        bar.addWidget(btn_export)
        lay.addLayout(bar)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        lay.addWidget(self.text, 1)
        self.level_combo.currentIndexChanged.connect(self.load_logs)
        self.keyword_edit.returnPressed.connect(self.load_logs)
        self.log_combo.currentIndexChanged.connect(self.load_logs)
        self.load_logs()

    def _filtered_lines(self):
        log_path = self.ctx.cm.logs_root / self.log_combo.currentData()
        if not log_path.exists():
            return []
        min_level = LEVELS[self.level_combo.currentText()]
        keyword = self.keyword_edit.text().strip().lower()
        out = []
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if min_level is not None:
                # 行内形如 [INFO]
                import re
                m = re.search(r"\[(DEBUG|INFO|WARNING|ERROR|CRITICAL)\]", line)
                if not m:
                    continue
                level = getattr(__import__("logging"), m.group(1))
                if level < min_level:
                    continue
            if keyword and keyword not in line.lower():
                continue
            out.append(line)
        return out

    def load_logs(self):
        lines = self._filtered_lines()
        self.text.setPlainText("\n".join(lines[-2000:]))

    def export_logs(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出日志", "logs_export.txt",
                                              "文本 (*.txt);;CSV (*.csv)")
        if not path:
            return
        lines = self._filtered_lines()
        if path.lower().endswith(".csv"):
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["log"])
                for line in lines:
                    writer.writerow([line])
        else:
            Path(path).write_text("\n".join(lines), encoding="utf-8")
        msg_info(self, "导出完成", f"已导出 {len(lines)} 行日志到：\n{path}")
