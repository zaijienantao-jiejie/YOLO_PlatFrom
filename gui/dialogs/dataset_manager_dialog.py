# -*- coding: utf-8 -*-
"""数据集管理对话框：统计信息、类别管理、划分、快照管理、验证报告"""
from pathlib import Path

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget,
                               QWidget, QMessageBox, QInputDialog, QListWidget,
                               QListWidgetItem, QGroupBox, QFormLayout)

from core.exceptions import DatasetError
from data.validation import validate_dataset
from gui.common import msg_error, msg_info, task_label


def show_validation_report(parent, ds, report: dict):
    """验证报告对话框"""
    dlg = QDialog(parent)
    dlg.setWindowTitle(f"验证报告 - {ds.name}")
    dlg.resize(760, 560)
    lay = QVBoxLayout(dlg)
    grp = QGroupBox("数据集验证报告")
    fl = QFormLayout(grp)
    fl.addRow("任务类型", QLabel(task_label(report["task"])))
    fl.addRow("总图像数", QLabel(str(report["total"])))
    fl.addRow("有效图像", QLabel(str(report["valid"])))
    fl.addRow("无效图像", QLabel(str(report["invalid"])))
    per_class = report.get("per_class", {})
    names = report.get("class_names", [])
    dist = ", ".join(f"{names[cid] if cid < len(names) else cid}: {n}" for cid, n in per_class.items())
    fl.addRow("类别分布", QLabel(dist or "无标注"))
    lay.addWidget(grp)
    table = QTableWidget(0, 3)
    table.setHorizontalHeaderLabels(["图像", "问题类型", "说明"])
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    issues = report.get("issues", [])
    table.setRowCount(len(issues))
    for i, issue in enumerate(issues):
        table.setItem(i, 0, QTableWidgetItem(issue.get("image", "")))
        table.setItem(i, 1, QTableWidgetItem(issue.get("type", "")))
        table.setItem(i, 2, QTableWidgetItem(issue.get("message", "")))
    lay.addWidget(table, 1)
    summary = QLabel(f"共发现 {len(issues)} 个问题" if issues else "✓ 未发现问题")
    summary.setObjectName("error" if issues else "success")
    lay.addWidget(summary)
    btn = QPushButton("关闭")
    btn.clicked.connect(dlg.accept)
    lay.addWidget(btn)
    dlg.exec()


class DatasetManagerDialog(QDialog):
    def __init__(self, ctx, ds, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.ds = ds
        self.setWindowTitle(f"数据集管理 - {ds.name}")
        self.resize(680, 560)
        lay = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self._build_info_tab(), "📊 统计")
        tabs.addTab(self._build_splits_tab(), "✂️ 数据划分")
        tabs.addTab(self._build_snapshot_tab(), "📸 快照")
        lay.addWidget(tabs)
        btn = QPushButton("关闭")
        btn.clicked.connect(self.accept)
        lay.addWidget(btn)
        self._refresh_info()

    def _build_info_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        grp = QGroupBox("统计信息")
        self.info_form = QFormLayout(grp)
        lay.addWidget(grp)
        self.class_table = QTableWidget(0, 3)
        self.class_table.setHorizontalHeaderLabels(["类别 ID", "名称", "颜色"])
        self.class_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        lay.addWidget(self.class_table, 1)
        return w

    def _refresh_info(self):
        st = self.ds.stats()
        while self.info_form.rowCount():
            self.info_form.removeRow(0)
        self.info_form.addRow("任务类型", QLabel(task_label(self.ds.task)))
        self.info_form.addRow("图像总数", QLabel(str(st["total"])))
        self.info_form.addRow("已标注", QLabel(str(st["labeled"])))
        self.info_form.addRow("未标注", QLabel(str(st["unlabeled"])))
        self.info_form.addRow("AI 标注", QLabel(str(st["ai_marked"])))
        self.info_form.addRow("版本", QLabel(f"v{st['version']}"))
        self.info_form.addRow("创建时间", QLabel(st["created_at"]))
        self.info_form.addRow("更新时间", QLabel(st["updated_at"]))
        self.class_table.setRowCount(0)
        for c in self.ds.meta.classes:
            row = self.class_table.rowCount()
            self.class_table.insertRow(row)
            self.class_table.setItem(row, 0, QTableWidgetItem(str(c.id)))
            self.class_table.setItem(row, 1, QTableWidgetItem(c.name))
            self.class_table.setItem(row, 2, QTableWidgetItem(c.color))

    def _build_splits_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        hint = QLabel("训练/验证/测试集划分。训练前自动执行，也可手动重新划分。")
        hint.setObjectName("dim")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        self.split_label = QLabel("")
        lay.addWidget(self.split_label)
        btn_split = QPushButton("✂️ 重新划分（随机种子）")
        btn_split.setObjectName("primary")
        btn_split.clicked.connect(self._resplit)
        lay.addWidget(btn_split)
        lay.addStretch(1)
        return w

    def _resplit(self):
        if self.ds.task == "classify":
            msg_info(self, "提示", "分类任务暂不支持数据划分（直接使用全部图像训练）")
            return
        try:
            counts = self.ds.generate_splits()
        except DatasetError as e:
            msg_error(self, "划分失败", str(e))
            return
        self.split_label.setText("  ".join(f"{k}: {v} 张" for k, v in counts.items()))
        self.ctx.activity.log("数据划分", self.ds.name, str(counts), actor=self.ctx.auth.current_username)
        msg_info(self, "划分完成", "训练/验证/测试集已重新生成")

    def _build_snapshot_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.snap_list = QListWidget()
        lay.addWidget(self.snap_list, 1)
        bar = QHBoxLayout()
        btn_create = QPushButton("📸 创建快照")
        btn_create.setObjectName("primary")
        btn_create.clicked.connect(self._create_snapshot)
        btn_restore = QPushButton("↩ 恢复所选快照")
        btn_restore.clicked.connect(self._restore_snapshot)
        bar.addWidget(btn_create)
        bar.addWidget(btn_restore)
        bar.addStretch(1)
        lay.addLayout(bar)
        self._refresh_snapshots()
        return w

    def _refresh_snapshots(self):
        self.snap_list.clear()
        for snap in self.ds.list_snapshots():
            self.snap_list.addItem(QListWidgetItem(snap.name))

    def _create_snapshot(self):
        note, ok = QInputDialog.getText(self, "创建快照", "快照说明（可选）：")
        if not ok:
            return
        self.ds.snapshot(note)
        self._refresh_snapshots()

    def _restore_snapshot(self):
        item = self.snap_list.currentItem()
        if item is None:
            msg_error(self, "未选择", "请选择要恢复的快照")
            return
        if QMessageBox.question(self, "确认恢复", "恢复快照会覆盖当前的标注文件，确定继续？") != QMessageBox.Yes:
            return
        snap_path = self.ds.snapshots_dir / item.text()
        self.ds.restore_snapshot(snap_path)
        self._refresh_info()
        msg_info(self, "恢复完成", "数据集已从快照恢复")
