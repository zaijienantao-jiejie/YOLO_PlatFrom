# -*- coding: utf-8 -*-
"""数据集管理工作流：5 种任务共用的基类（列表 + 新建/导入/标注/验证/增强/导出/快照）"""
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
                               QDialog, QDialogButtonBox, QLineEdit, QTextEdit, QComboBox,
                               QSpinBox, QDoubleSpinBox, QProgressDialog, QMessageBox,
                               QAbstractItemView, QInputDialog, QFormLayout, QGroupBox)

from core import logger
from core.exceptions import DatasetError
from data.augmentation import Augmenter, STRATEGIES
from data.validation import validate_dataset
from gui.common import task_label, task_icon, msg_error, msg_info, msg_confirm, form_row
from gui.workers import FunctionWorker


class RefreshWorker(QThread):
    """后台计算数据集统计，避免大数据集时界面卡顿"""
    row = Signal(dict)
    finished = Signal()

    def __init__(self, dm, task, parent=None):
        super().__init__(parent)
        self._dm = dm
        self._task = task

    def run(self):
        for ds in self._dm.list():
            if ds.task != self._task:
                continue
            try:
                st = ds.stats()
                self.row.emit({
                    "name": ds.name, "task": ds.task, "total": st["total"],
                    "labeled": st["labeled"], "ai": st["ai_marked"],
                    "updated": st["updated_at"] or "-",
                })
            except Exception:
                continue
        self.finished.emit()


class DatasetCreateDialog(QDialog):
    """新建数据集对话框"""

    def __init__(self, task: str, parent=None):
        super().__init__(parent)
        self.task = task
        self.setWindowTitle(f"新建数据集 - {task_label(task)}")
        lay = QVBoxLayout(self)
        self.name_edit = QLineEdit()
        lay.addWidget(form_row("数据集名称", self.name_edit))
        self.desc_edit = QLineEdit()
        lay.addWidget(form_row("描述", self.desc_edit))
        lay.addWidget(QLabel("类别（每行一个）："))
        self.classes_edit = QTextEdit()
        self.classes_edit.setPlaceholderText("person\ncar\n...")
        lay.addWidget(self.classes_edit)
        if task == "pose":
            lay.addWidget(QLabel("关键点（每行一个，如 nose / left_eye / right_eye）："))
            self.kpts_edit = QTextEdit()
            self.kpts_edit.setPlaceholderText("nose\nleft_eye\nright_eye")
            lay.addWidget(self.kpts_edit)
        grp = QGroupBox("数据集划分比例（训练/验证/测试）")
        fl = QFormLayout(grp)
        self.train_spin = QDoubleSpinBox()
        self.train_spin.setRange(0.1, 0.9)
        self.train_spin.setValue(0.7)
        self.train_spin.setSingleStep(0.05)
        self.val_spin = QDoubleSpinBox()
        self.val_spin.setRange(0.0, 0.5)
        self.val_spin.setValue(0.2)
        self.val_spin.setSingleStep(0.05)
        self.test_spin = QDoubleSpinBox()
        self.test_spin.setRange(0.0, 0.5)
        self.test_spin.setValue(0.1)
        self.test_spin.setSingleStep(0.05)
        fl.addRow("训练", self.train_spin)
        fl.addRow("验证", self.val_spin)
        fl.addRow("测试", self.test_spin)
        lay.addWidget(grp)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def get_classes(self):
        return [l.strip() for l in self.classes_edit.toPlainText().splitlines() if l.strip()]

    def get_keypoints(self):
        if self.task != "pose":
            return None
        return [l.strip() for l in getattr(self, "kpts_edit").toPlainText().splitlines() if l.strip()]

    def get_split_ratio(self):
        return {"train": self.train_spin.value(), "val": self.val_spin.value(), "test": self.test_spin.value()}


class DatasetWorkflowWidget(QWidget):
    """数据集管理工作流基类；子类只需设置 self.task"""
    task = "detect"
    datasets_changed = None

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._refresh_worker = None
        self._pending_refresh = False
        self._rows = []
        self._task_workers = []
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        root = QVBoxLayout(self)
        title = QLabel(f"{task_icon(self.task)} {task_label(self.task)} 数据集")
        title.setObjectName("title")
        root.addWidget(title)

        bar = QHBoxLayout()
        self.btn_new = QPushButton("➕ 新建数据集")
        self.btn_new.setObjectName("primary")
        self.btn_new.clicked.connect(self.create_dataset)
        self.btn_import = QPushButton("📥 导入数据集")
        self.btn_import.clicked.connect(self.import_dataset)
        self.btn_annotate = QPushButton("✏️ 打开标注")
        self.btn_annotate.clicked.connect(self.open_annotator)
        self.btn_validate = QPushButton("🔍 验证")
        self.btn_validate.clicked.connect(self.validate_selected)
        self.btn_augment = QPushButton("✨ 数据增强")
        self.btn_augment.clicked.connect(self.augment_selected)
        self.btn_export = QPushButton("📤 导出")
        self.btn_export.clicked.connect(self.export_selected)
        self.btn_snapshot = QPushButton("📸 快照")
        self.btn_snapshot.clicked.connect(self.snapshot_selected)
        self.btn_copy = QPushButton("📋 复制")
        self.btn_copy.clicked.connect(self.copy_selected)
        self.btn_rename = QPushButton("✏️ 重命名")
        self.btn_rename.clicked.connect(self.rename_selected)
        self.btn_delete = QPushButton("🗑 删除")
        self.btn_delete.setObjectName("danger")
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_refresh = QPushButton("🔄 刷新")
        self.btn_refresh.clicked.connect(self.refresh)
        for b in (self.btn_new, self.btn_import, self.btn_annotate, self.btn_validate,
                  self.btn_augment, self.btn_export, self.btn_snapshot, self.btn_copy,
                  self.btn_rename, self.btn_delete, self.btn_refresh):
            bar.addWidget(b)
        bar.addStretch(1)
        root.addLayout(bar)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["名称", "任务", "图像数", "已标注", "AI 标注", "更新时间"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.doubleClicked.connect(self.open_annotator)
        root.addWidget(self.table, 1)

        hint = QLabel("提示：双击数据集直接进入标注；训练前请在“验证”中检查数据质量。")
        hint.setObjectName("dim")
        root.addWidget(hint)

    # ---------------- 刷新（后台线程计算统计） ----------------
    def refresh(self):
        if self._refresh_worker is not None and self._refresh_worker.isRunning():
            self._pending_refresh = True
            return
        self._rows = []
        self._refresh_worker = RefreshWorker(self.ctx.dm, self.task, self)
        self._refresh_worker.row.connect(self._rows.append)
        self._refresh_worker.finished.connect(self._on_refresh_done)
        self._refresh_worker.start()

    def _on_refresh_done(self):
        self.table.setRowCount(0)
        for data in self._rows:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(data["name"]))
            self.table.setItem(row, 1, QTableWidgetItem(task_label(data["task"])))
            self.table.setItem(row, 2, QTableWidgetItem(str(data["total"])))
            self.table.setItem(row, 3, QTableWidgetItem(str(data["labeled"])))
            self.table.setItem(row, 4, QTableWidgetItem(str(data["ai"])))
            self.table.setItem(row, 5, QTableWidgetItem(data["updated"]))
        if self._pending_refresh:
            self._pending_refresh = False
            self.refresh()

    # ---------------- 通用后台任务 ----------------
    def _run_task(self, title: str, task_fn, on_success, indeterminate: bool = False):
        """耗时操作放入后台线程，显示无取消按钮的进度对话框"""
        prog = QProgressDialog(f"{title} ...", "", 0, 100, self)
        prog.setCancelButton(None)
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(300)
        if indeterminate:
            prog.setRange(0, 0)  # 忙碌指示
        worker = FunctionWorker(task_fn)
        worker.progress.connect(lambda d, t: (prog.setMaximum(max(t, 1)), prog.setValue(d)))
        worker.succeeded.connect(lambda result: (prog.close(), on_success(result)))
        worker.failed.connect(lambda m: (prog.close(), msg_error(self, "操作失败", m)))
        self._task_workers = [w for w in self._task_workers if w.isRunning()]
        self._task_workers.append(worker)
        worker.start()
        return worker

    def _selected_dataset(self):
        row = self.table.currentRow()
        if row < 0:
            msg_error(self, "未选择", "请先在列表中选择一个数据集")
            return None
        name = self.table.item(row, 0).text()
        return self.ctx.dm.get(name)

    # ---------------- 操作 ----------------
    def create_dataset(self):
        dlg = DatasetCreateDialog(self.task, self)
        if dlg.exec() != QDialog.Accepted:
            return
        classes = dlg.get_classes()
        if not classes:
            msg_error(self, "参数错误", "请填写至少 1 个类别")
            return
        try:
            ds = self.ctx.dm.create(
                name=dlg.name_edit.text().strip(), task=self.task, classes=classes,
                description=dlg.desc_edit.text().strip(),
                keypoints=dlg.get_keypoints(), split_ratio=dlg.get_split_ratio())
        except DatasetError as e:
            msg_error(self, "创建失败", str(e))
            return
        self.ctx.activity.log("创建数据集", ds.name, f"任务: {self.task}, 类别: {len(classes)}",
                              actor=self.ctx.auth.current_username)
        msg_info(self, "创建成功", f"数据集 {ds.name} 已创建。\n\n下一步：导入图像（在标注界面中点击“添加图像”，或从标注编辑器导入）。")
        self.refresh()

    def import_dataset(self):
        dlg = DatasetImportDialog(self.ctx, self.task, self)
        if dlg.exec() == QDialog.Accepted:
            self.refresh()

    def open_annotator(self):
        ds = self._selected_dataset()
        if ds is None:
            return
        from gui.widgets.annotation_editor import AnnotationEditor
        editor = AnnotationEditor(self.ctx, ds)
        editor.resize(1500, 950)
        editor.show()
        # 保存引用，防止被垃圾回收
        if not hasattr(self, "_editors"):
            self._editors = []
        self._editors = [e for e in getattr(self, "_editors", []) if e.isVisible()]
        self._editors.append(editor)

    def validate_selected(self):
        ds = self._selected_dataset()
        if ds is None:
            return

        def task(cb):
            return validate_dataset(ds, progress_cb=cb)

        def on_success(report):
            from gui.dialogs.dataset_manager_dialog import show_validation_report
            show_validation_report(self, ds, report)

        self._run_task(f"正在验证数据集 {ds.name}", task, on_success)

    def augment_selected(self):
        ds = self._selected_dataset()
        if ds is None:
            return
        if ds.task == "classify":
            msg_info(self, "提示", "分类任务暂不支持离线增强（训练时会自动应用在线增强）")
            return
        from PySide6.QtWidgets import QDialog as QD
        dlg = QD(self)
        dlg.setWindowTitle(f"数据增强 - {ds.name}")
        lay = QVBoxLayout(dlg)
        lay.addWidget(form_row("增强策略", (combo := QComboBox())))
        for key, label in STRATEGIES.items():
            combo.addItem(label, key)
        if ds.task == "pose":
            # 姿态任务不支持 mosaic
            idx = combo.findData("mosaic")
            combo.model().item(idx).setEnabled(False)
        combo.setCurrentIndex(0)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)
        if dlg.exec() != QD.Accepted:
            return
        strategy = combo.currentData()

        def task(cb):
            return Augmenter(ds).apply(strategy, progress_cb=cb)

        def on_success(n):
            self.ctx.activity.log("数据增强", ds.name, f"{STRATEGIES[strategy]} ×{n}",
                                  actor=self.ctx.auth.current_username)
            msg_info(self, "增强完成", f"已生成 {n} 张增强图像")
            self.refresh()

        self._run_task(f"正在增强数据集 {ds.name}", task, on_success)

    def export_selected(self):
        ds = self._selected_dataset()
        if ds is None:
            return
        from PySide6.QtWidgets import QDialog as QD
        dlg = QD(self)
        dlg.setWindowTitle(f"导出数据集 - {ds.name}")
        lay = QVBoxLayout(dlg)
        lay.addWidget(form_row("导出格式", (combo := QComboBox())))
        for fmt, label in (("yolo", "YOLO 格式"), ("coco", "COCO 格式"), ("voc", "Pascal VOC 格式")):
            combo.addItem(label, fmt)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)
        if dlg.exec() != QD.Accepted:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "选择导出目录", str(self.ctx.cm.outputs_root))
        if not out_dir:
            return
        fmt = combo.currentData()

        def task(cb):
            return self.ctx.dm.export(ds, fmt, Path(out_dir))

        def on_success(result):
            self.ctx.activity.log("导出数据集", ds.name, f"格式: {fmt}",
                                  actor=self.ctx.auth.current_username)
            msg_info(self, "导出成功", f"已导出到：\n{result}")

        self._run_task(f"正在导出数据集 {ds.name}", task, on_success, indeterminate=True)

    def snapshot_selected(self):
        ds = self._selected_dataset()
        if ds is None:
            return
        note, ok = QInputDialog.getText(self, "创建快照", "快照说明（可选）：")
        if not ok:
            return
        zip_path = ds.snapshot(note)
        self.ctx.activity.log("数据集快照", ds.name, zip_path.name, actor=self.ctx.auth.current_username)
        msg_info(self, "快照完成", f"快照已保存：\n{zip_path.name}")

    def copy_selected(self):
        ds = self._selected_dataset()
        if ds is None:
            return
        new_name, ok = QInputDialog.getText(self, "复制数据集", "新数据集名称：", text=f"{ds.name}_副本")
        if not ok or not new_name.strip():
            return
        try:
            import shutil
            dst = self.ctx.dm.root / new_name.strip()
            if dst.exists():
                raise DatasetError(f"数据集 {new_name.strip()} 已存在")
            shutil.copytree(ds.root, dst)
            new_ds = self.ctx.dm.get(new_name.strip())
            new_ds.meta.name = new_name.strip()
            new_ds._touch()
        except DatasetError as e:
            msg_error(self, "复制失败", str(e))
            return
        self.ctx.activity.log("复制数据集", f"{ds.name} → {new_name.strip()}", "",
                              actor=self.ctx.auth.current_username)
        msg_info(self, "复制完成", f"数据集已复制为 {new_name.strip()}")
        self.refresh()

    def rename_selected(self):
        ds = self._selected_dataset()
        if ds is None:
            return
        new_name, ok = QInputDialog.getText(self, "重命名数据集", "新名称：", text=ds.name)
        if not ok or not new_name.strip() or new_name.strip() == ds.name:
            return
        try:
            self.ctx.dm.rename(ds.name, new_name.strip())
        except DatasetError as e:
            msg_error(self, "重命名失败", str(e))
            return
        self.ctx.activity.log("重命名数据集", f"{ds.name} → {new_name.strip()}", "",
                              actor=self.ctx.auth.current_username)
        self.refresh()

    def delete_selected(self):
        ds = self._selected_dataset()
        if ds is None:
            return
        if not msg_confirm(self, "确认删除",
                           f"确定删除数据集 {ds.name}？\n（包含 {len(ds.image_files())} 张图像，删除后不可恢复，建议先创建快照）"):
            return
        try:
            self.ctx.dm.delete(ds.name)
        except DatasetError as e:
            msg_error(self, "删除失败", str(e))
            return
        self.ctx.activity.log("删除数据集", ds.name, "", actor=self.ctx.auth.current_username)
        self.refresh()


class DatasetImportDialog(QDialog):
    """导入数据集对话框"""

    def __init__(self, ctx, task: str, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.task = task
        self.setWindowTitle(f"导入数据集 - {task_label(task)}")
        lay = QVBoxLayout(self)
        self.name_edit = QLineEdit()
        lay.addWidget(form_row("数据集名称", self.name_edit))
        self.src_row = QWidget()
        lay.addWidget(form_row("数据源", self.src_row))
        sr = QHBoxLayout(self.src_row)
        sr.setContentsMargins(0, 0, 0, 0)
        self.src_edit = QLineEdit()
        sr.addWidget(self.src_edit, 1)
        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self._browse)
        sr.addWidget(btn_browse)
        self.fmt_combo = QComboBox()
        lay.addWidget(form_row("源格式", self.fmt_combo))
        if task == "classify":
            self.fmt_combo.addItem("文件夹分类（每类一个子目录，支持 train/val 分目录）", "folder")
        else:
            self.fmt_combo.addItem("YOLO 格式（images/ + labels/）", "yolo")
            self.fmt_combo.addItem("COCO 格式（annotations.json）", "coco")
            self.fmt_combo.addItem("Pascal VOC 格式（Annotations/ + JPEGImages/）", "voc")
        self.classes_edit = QLineEdit()
        lay.addWidget(form_row("类别（可选，留空自动识别）", self.classes_edit))
        self.classes_edit.setPlaceholderText("如: person,car,cat")
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._do_import)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _browse(self):
        if self.task == "classify" or self.fmt_combo.currentData() == "coco":
            path = QFileDialog.getOpenFileName(self, "选择数据源")[0] if self.fmt_combo.currentData() == "coco" \
                else QFileDialog.getExistingDirectory(self, "选择分类文件夹")
        else:
            path = QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if path:
            self.src_edit.setText(path)

    def _do_import(self):
        src = Path(self.src_edit.text().strip())
        name = self.name_edit.text().strip()
        if not name:
            msg_error(self, "参数错误", "请填写数据集名称")
            return
        if not src.exists():
            msg_error(self, "参数错误", "数据源路径不存在")
            return
        classes = [c.strip() for c in self.classes_edit.text().split(",") if c.strip()] or None
        fmt = self.fmt_combo.currentData()
        prog = QProgressDialog("正在导入数据集 ...", "", 0, 100, self)
        prog.setCancelButton(None)
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(300)

        worker = FunctionWorker(
            lambda cb: self.ctx.dm.import_from(src, fmt, name, self.task, class_names=classes, progress_cb=cb))
        worker.progress.connect(lambda d, t: (prog.setMaximum(max(t, 1)), prog.setValue(d)))
        worker.succeeded.connect(lambda ds: self._on_import_done(prog, ds, src, fmt))
        worker.failed.connect(lambda m: (prog.close(), msg_error(self, "导入失败", m)))
        self._import_worker = worker
        worker.start()

    def _on_import_done(self, prog, ds, src, fmt):
        prog.close()
        self.ctx.activity.log("导入数据集", ds.name, f"来源: {src}, 格式: {fmt}",
                              actor=self.ctx.auth.current_username)
        msg_info(self, "导入成功", f"数据集 {ds.name} 导入完成，共 {len(ds.image_files())} 张图像")
        self.accept()
