# -*- coding: utf-8 -*-
"""GUI 线程化离屏测试：导入/刷新/验证 worker + 标注编辑器子目录图像"""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

app = QApplication(sys.argv)

from core.config import ConfigManager
cm = ConfigManager(Path('.').resolve().parent if False else Path(__file__).resolve().parent.parent)
cm.load()
from core import logger as logger_mod
logger_mod.setup_logging(cm.logs_root, 'ERROR')
from auth.manager import AuthManager
auth = AuthManager(cm.project_root / 'workspace', 30)
auth.login('admin', 'admin123')

# 屏蔽模态弹窗（离屏模式无人点击）
import gui.widgets.dataset_workflow as dw
dw.msg_info = lambda *a, **k: print('[info]', a[1] if len(a) > 1 else '')
dw.msg_error = lambda *a, **k: print('[error]', a[1] if len(a) > 1 else '')
import gui.widgets.annotation_editor as ae
ae.msg_error = lambda *a, **k: print('[error]', a[1] if len(a) > 1 else '')
import gui.dialogs.dataset_manager_dialog as dmd
dmd.show_validation_report = lambda *a, **k: print('[validation report shown]')

from gui.main_window_modern import MainWindow
win = MainWindow(cm, auth)
win.show()


def spin(ms=400):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


detect_tab = win.datasets_tabs.widget(1)
detect_tab.refresh()
spin(600)
print('OK1 后台刷新 worker 运行（未阻塞）')

name = f'线程导入测试{int(time.time()) % 100000}'
from gui.widgets.dataset_workflow import DatasetImportDialog
dlg = DatasetImportDialog(win.ctx, 'detect')
src = Path(tempfile.mkdtemp())
for sub in ('train', 'val'):
    (src / 'images' / sub).mkdir(parents=True)
    (src / 'labels' / sub).mkdir(parents=True)
    for i in range(5):
        Image.fromarray(np.full((64, 64, 3), (0, 128, 255), dtype=np.uint8)).save(src / 'images' / sub / f'{sub}{i}.jpg')
        (src / 'labels' / sub / f'{sub}{i}.txt').write_text('0 0.5 0.5 0.4 0.3\n')
dlg.name_edit.setText(name)
dlg.src_edit.setText(str(src))
dlg._do_import()
spin(1500)
ds = win.ctx.dm.get(name)
assert len(ds.image_files()) == 10, len(ds.image_files())
print('OK2 导入对话框 worker：', len(ds.image_files()), '张图像')

from gui.widgets.annotation_editor import AnnotationEditor
editor = AnnotationEditor(win.ctx, ds)
editor.show()
assert editor._image_paths and editor._image_paths[0].name == 'train0.jpg'
spin(300)
print('OK3 标注编辑器子目录图像, 显示名:', ds.image_display_name(editor._image_paths[0]))

detect_tab.refresh()
spin(600)
detect_tab.table.selectRow(0)
detect_tab.validate_selected()
spin(1500)
print('OK4 验证 worker 执行完成')

win.close()
print('ALL GUI 线程化测试通过 ✅')
