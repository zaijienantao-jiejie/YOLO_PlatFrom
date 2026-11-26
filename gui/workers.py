# -*- coding: utf-8 -*-
"""通用后台任务 worker：把耗时操作（导入/验证/增强/导出等）移出 GUI 线程，避免界面未响应"""
import traceback

from PySide6.QtCore import QThread, Signal


class FunctionWorker(QThread):
    """在后台线程执行任意函数。

    函数签名: fn(progress_cb) -> result
      progress_cb(done, total) 用于上报进度（可选）
    """
    progress = Signal(int, int)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, with_progress: bool = True, parent=None):
        super().__init__(parent)
        self._fn = fn
        self._with_progress = with_progress

    def run(self):
        try:
            cb = (lambda done, total: self.progress.emit(done, total)) if self._with_progress else None
            result = self._fn(cb)
            self.succeeded.emit(result)
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(str(e))
