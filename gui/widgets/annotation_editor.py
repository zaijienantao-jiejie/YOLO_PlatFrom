# -*- coding: utf-8 -*-
"""交互式数据标注编辑器：矩形框 / 多边形 / 关键点 / 旋转框 / 类别标签

包含 5 种任务的画布实现与编辑器外壳（图像列表 + 画布 + 标注控制面板）。
"""
import copy
import math
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import (QColor, QPainterPath, QPen, QBrush, QPainter)
from PySide6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                               QGraphicsPathItem, QGraphicsEllipseItem, QGraphicsLineItem,
                               QWidget, QListWidget, QListWidgetItem, QTableWidget,
                               QTableWidgetItem, QComboBox, QLabel, QPushButton,
                               QSplitter, QVBoxLayout, QHBoxLayout, QAbstractItemView)

from core import logger
from gui.common import load_pixmap, task_label, msg_error


class BaseCanvas(QGraphicsView):
    """标注画布基类：图像显示、缩放、坐标换算、撤销/重做、选中"""
    ann_changed = Signal()

    def __init__(self, ds, editor, parent=None):
        super().__init__(parent)
        self.ds = ds
        self.editor = editor
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.NoDrag)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._img_w = self._img_h = 0
        self.anns = []            # 内部标注列表（唯一数据源）
        self.selected_index = None
        self._undo_stack = []
        self._redo_stack = []
        self._drawing = None      # 绘制中的临时数据
        self._preview_items = []  # 预览图形
        self._ann_items = []      # 标注图形（重建时只移除它们，保留底图）
        self.point_threshold = 8  # 像素

    # ---------------- 图像 ----------------
    def set_image(self, path: Path):
        self._scene.clear()
        self._ann_items.clear()
        self._preview_items.clear()
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        pm = load_pixmap(path, max_size=(3000, 2000))
        self._pixmap_item.setPixmap(pm)
        self._scene.setSceneRect(QRectF(pm.rect()))
        self._img_w = pm.width()
        self._img_h = pm.height()
        self.fit_image()
        self.anns = []
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.selected_index = None

    def fit_image(self):
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        # 限制最小缩放，避免过度放大
        self.scale(1.0, 1.0)

    def wheelEvent(self, ev):
        factor = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def clear_annotations(self):
        self.anns = []
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.rebuild()

    # ---------------- 坐标 ----------------
    def norm_to_pix(self, x, y):
        return QPointF(x * self._img_w, y * self._img_h)

    def pix_to_norm(self, p: QPointF):
        if self._img_w == 0 or self._img_h == 0:
            return (0.0, 0.0)
        return (max(0.0, min(1.0, p.x() / self._img_w)), max(0.0, min(1.0, p.y() / self._img_h)))

    def _dist(self, p1: QPointF, p2: QPointF) -> float:
        return math.hypot(p1.x() - p2.x(), p1.y() - p2.y())

    def _point_in_rect(self, p: QPointF, r: QRectF) -> bool:
        return r.adjusted(-self.point_threshold, -self.point_threshold,
                          self.point_threshold, self.point_threshold).contains(p)

    # ---------------- 撤销 / 重做 ----------------
    def snapshot(self):
        self._undo_stack.append(copy.deepcopy(self.anns))
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self):
        if not self._undo_stack:
            return
        self._redo_stack.append(copy.deepcopy(self.anns))
        self.anns = self._undo_stack.pop()
        self.selected_index = None
        self.rebuild()
        self.ann_changed.emit()

    def redo(self):
        if not self._redo_stack:
            return
        self._undo_stack.append(copy.deepcopy(self.anns))
        self.anns = self._redo_stack.pop()
        self.selected_index = None
        self.rebuild()
        self.ann_changed.emit()

    def delete_selected(self):
        if self.selected_index is None:
            return
        self.snapshot()
        del self.anns[self.selected_index]
        self.selected_index = None
        self.rebuild()
        self.ann_changed.emit()

    # ---------------- 图形重建 ----------------
    def rebuild(self):
        raise NotImplementedError

    def _commit_ann(self, ann: dict):
        self.snapshot()
        self.anns.append(ann)
        self.rebuild()
        self.ann_changed.emit()

    def set_class_id(self, cid: int):
        self._current_class = cid

    def to_annotations(self):
        return copy.deepcopy(self.anns)

    def load_annotations(self, anns):
        self.anns = copy.deepcopy(anns)
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.rebuild()

    def _clear_preview(self):
        for item in self._preview_items:
            self._scene.removeItem(item)
        self._preview_items.clear()

    def _reset_ann_items(self):
        """移除全部标注图形（保留底图）"""
        for item in self._ann_items:
            self._scene.removeItem(item)
        self._ann_items.clear()

    def _selected_color(self, cid: int) -> QColor:
        cls = self.ds.class_by_id(cid)
        c = QColor(cls.color if cls and cls.color else "#4f8cff")
        return c


# ================================================================ 目标检测
class DetectCanvas(BaseCanvas):
    def rebuild(self):
        self._reset_ann_items()
        for i, ann in enumerate(self.anns):
            cx, cy, w, h = ann["bbox"]
            rect = QRectF((cx - w / 2) * self._img_w, (cy - h / 2) * self._img_h, w * self._img_w, h * self._img_h)
            color = self._selected_color(ann["class_id"])
            pen = QPen(color, 3 if i == self.selected_index else 2)
            item = self._scene.addRect(rect, pen)
            item.setData(0, i)
            self._ann_items.append(item)
            if i == self.selected_index:
                # 四角手柄
                for cx0, cy0 in ((rect.left(), rect.top()), (rect.right(), rect.top()),
                                 (rect.left(), rect.bottom()), (rect.right(), rect.bottom())):
                    h = self._scene.addEllipse(cx0 - 4, cy0 - 4, 8, 8, QPen(color), QBrush(color))
                    h.setData(0, i)
                    self._ann_items.append(h)

    def _hit_test(self, p: QPointF):
        for i in range(len(self.anns) - 1, -1, -1):
            cx, cy, w, h = self.anns[i]["bbox"]
            rect = QRectF((cx - w / 2) * self._img_w, (cy - h / 2) * self._img_h, w * self._img_w, h * self._img_h)
            if self._point_in_rect(p, rect):
                corners = [(rect.left(), rect.top()), (rect.right(), rect.top()),
                           (rect.left(), rect.bottom()), (rect.right(), rect.bottom())]
                for k, (x, y) in enumerate(corners):
                    if self._dist(p, QPointF(x, y)) <= self.point_threshold * 1.5:
                        return i, "resize", k
                return i, "move", None
        return None, None, None

    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            return super().mousePressEvent(ev)
        p = self.mapToScene(ev.position().toPoint())
        idx, mode, corner = self._hit_test(p)
        if idx is not None:
            self.snapshot()
            self.selected_index = idx
            self._drawing = {"mode": mode, "index": idx, "corner": corner, "start": p}
            self.rebuild()
        else:
            self.selected_index = None
            self._drawing = {"mode": "new", "start": p}
            self.rebuild()
        self.ann_changed.emit()

    def mouseMoveEvent(self, ev):
        if self._drawing is None:
            return
        p = self.mapToScene(ev.position().toPoint())
        mode = self._drawing["mode"]
        if mode == "new":
            x0, y0 = self._drawing["start"].x(), self._drawing["start"].y()
            rect = QRectF(x0, y0, p.x() - x0, p.y() - y0).normalized()
            self._clear_preview()
            self._preview_items.append(self._scene.addRect(rect,
                QPen(QColor(self._selected_color(self._current_class)), 2, Qt.DashLine)))
        elif mode == "move":
            dx = (p.x() - self._drawing["start"].x()) / self._img_w
            dy = (p.y() - self._drawing["start"].y()) / self._img_h
            ann = self.anns[self._drawing["index"]]
            ann["bbox"][0] = min(1.0, max(0.0, ann["bbox"][0] + dx))
            ann["bbox"][1] = min(1.0, max(0.0, ann["bbox"][1] + dy))
            self.rebuild()
        elif mode == "resize":
            ann = self.anns[self._drawing["index"]]
            cx, cy, w, h = ann["bbox"]
            x0 = (cx - w / 2) * self._img_w
            y0 = (cy - h / 2) * self._img_h
            x1 = (cx + w / 2) * self._img_w
            y1 = (cy + h / 2) * self._img_h
            k = self._drawing["corner"]
            if k == 0: x0 = p.x(); y0 = p.y()
            elif k == 1: x1 = p.x(); y0 = p.y()
            elif k == 2: x0 = p.x(); y1 = p.y()
            else: x1 = p.x(); y1 = p.y()
            if x1 - x0 > 4 and y1 - y0 > 4:
                ann["bbox"] = [((x0 + x1) / 2) / self._img_w, ((y0 + y1) / 2) / self._img_h,
                               (x1 - x0) / self._img_w, (y1 - y0) / self._img_h]
            self.rebuild()

    def mouseReleaseEvent(self, ev):
        if ev.button() != Qt.LeftButton or self._drawing is None:
            return
        p = self.mapToScene(ev.position().toPoint())
        mode = self._drawing["mode"]
        self._clear_preview()
        if mode == "new":
            x0, y0 = self._drawing["start"].x(), self._drawing["start"].y()
            rect = QRectF(x0, y0, p.x() - x0, p.y() - y0).normalized()
            if rect.width() > 6 and rect.height() > 6 and self._img_w > 0:
                ann = {"class_id": self._current_class,
                       "bbox": [rect.center().x() / self._img_w, rect.center().y() / self._img_h,
                                rect.width() / self._img_w, rect.height() / self._img_h]}
                self._commit_ann(ann)
        self._drawing = None


# ================================================================ 实例分割
class SegmentCanvas(BaseCanvas):
    def rebuild(self):
        self._reset_ann_items()
        for i, ann in enumerate(self.anns):
            color = self._selected_color(ann["class_id"])
            path = QPainterPath()
            pts = [self.norm_to_pix(x, y) for x, y in ann["points"]]
            if pts:
                path.moveTo(pts[0])
                for pt in pts[1:]:
                    path.lineTo(pt)
                path.closeSubpath()
                pen = QPen(color, 3 if i == self.selected_index else 2)
                fill = QColor(color)
                fill.setAlpha(60)
                item = self._scene.addPath(path, pen, fill)
                item.setData(0, i)
                self._ann_items.append(item)

    def _hit_test(self, p: QPointF):
        from PySide6.QtGui import QPolygonF
        for i in range(len(self.anns) - 1, -1, -1):
            poly = QPolygonF([self.norm_to_pix(x, y) for x, y in self.anns[i]["points"]])
            if poly.containsPoint(p, Qt.OddEvenFill):
                return i
        return None

    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            return
        p = self.mapToScene(ev.position().toPoint())
        if self._drawing and self._drawing["mode"] == "polygon":
            # 绘制多边形中：点击添加顶点
            self._drawing["points"].append(p)
            self._update_polygon_preview()
            return
        idx = self._hit_test(p)
        if idx is not None:
            self.snapshot()
            self.selected_index = idx
            self._drawing = {"mode": "move_polygon", "index": idx, "start": p}
            self.rebuild()
            self.ann_changed.emit()
        else:
            self.selected_index = None
            self._drawing = {"mode": "polygon", "points": [p]}
            self._update_polygon_preview()
            self.rebuild()

    def _update_polygon_preview(self):
        self._clear_preview()
        pts = self._drawing["points"]
        color = self._selected_color(self._current_class)
        for k in range(len(pts) - 1):
            self._preview_items.append(self._scene.addLine(pts[k].x(), pts[k].y(),
                pts[k + 1].x(), pts[k + 1].y(), QPen(color, 2, Qt.DashLine)))
        for pt in pts:
            self._preview_items.append(self._scene.addEllipse(pt.x() - 3, pt.y() - 3, 6, 6, QPen(color), QBrush(color)))

    def mouseMoveEvent(self, ev):
        if self._drawing is None:
            return
        p = self.mapToScene(ev.position().toPoint())
        if self._drawing["mode"] == "polygon":
            pts = self._drawing["points"] + [p]
            self._clear_preview()
            color = self._selected_color(self._current_class)
            for k in range(len(pts) - 1):
                self._preview_items.append(self._scene.addLine(pts[k].x(), pts[k].y(),
                    pts[k + 1].x(), pts[k + 1].y(), QPen(color, 2, Qt.DashLine)))
            for pt in self._drawing["points"]:
                self._preview_items.append(self._scene.addEllipse(pt.x() - 3, pt.y() - 3, 6, 6, QPen(color), QBrush(color)))
        elif self._drawing["mode"] == "move_polygon":
            dx = (p.x() - self._drawing["start"].x()) / self._img_w
            dy = (p.y() - self._drawing["start"].y()) / self._img_h
            ann = self.anns[self._drawing["index"]]
            ann["points"] = [[min(1.0, max(0.0, x + dx)), min(1.0, max(0.0, y + dy))] for x, y in ann["points"]]
            self.rebuild()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.RightButton and self._drawing and self._drawing["mode"] == "polygon":
            self._finish_polygon()
            return
        if ev.button() != Qt.LeftButton or self._drawing is None:
            return
        if self._drawing["mode"] == "move_polygon":
            self._drawing = None
            self.rebuild()
            self.ann_changed.emit()

    def keyPressEvent(self, ev):
        if self._drawing and self._drawing["mode"] == "polygon" and ev.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._finish_polygon()
        elif self._drawing and self._drawing["mode"] == "polygon" and ev.key() == Qt.Key_Escape:
            self._drawing = None
            self._clear_preview()
        else:
            super().keyPressEvent(ev)

    def _finish_polygon(self):
        if self._drawing["mode"] != "polygon" or len(self._drawing["points"]) < 3:
            self._drawing = None
            self._clear_preview()
            return
        pts = [self.pix_to_norm(p) for p in self._drawing["points"]]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ann = {"class_id": self._current_class, "points": pts,
               "bbox": [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]}
        self._drawing = None
        self._clear_preview()
        self._commit_ann(ann)


# ================================================================ 姿态估计
class PoseCanvas(BaseCanvas):
    def rebuild(self):
        self._reset_ann_items()
        kpt_names = [k.name for k in self.ds.meta.keypoints]
        for i, ann in enumerate(self.anns):
            color = self._selected_color(ann["class_id"])
            kps = ann["keypoints"]
            pts = [self.norm_to_pix(x, y) for x, y, v in kps]
            # 骨架连线
            if self.ds.meta.skeleton:
                for a, b in self.ds.meta.skeleton:
                    if a < len(pts) and b < len(pts) and kps[a][2] > 0 and kps[b][2] > 0:
                        line = self._scene.addLine(pts[a].x(), pts[a].y(), pts[b].x(), pts[b].y(), QPen(color, 2))
                        line.setData(0, i)
                        self._ann_items.append(line)
            for k, pt in enumerate(pts):
                v = kps[k][2]
                r = 5 if v == 2 else 3
                c = color if v > 0 else QColor("#888888")
                e = self._scene.addEllipse(pt.x() - r, pt.y() - r, r * 2, r * 2, QPen(c), QBrush(c))
                e.setData(0, i)
                e.setData(1, k)
                self._ann_items.append(e)

    def _hit_test(self, p: QPointF):
        for i in range(len(self.anns) - 1, -1, -1):
            for k, (x, y, v) in enumerate(self.anns[i]["keypoints"]):
                pt = self.norm_to_pix(x, y)
                if self._dist(p, pt) <= self.point_threshold * 1.5:
                    return i, k
        return None, None

    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            return
        p = self.mapToScene(ev.position().toPoint())
        idx, kpt = self._hit_test(p)
        if idx is not None:
            self.snapshot()
            self.selected_index = idx
            self._drawing = {"mode": "move_kpt", "index": idx, "kpt": kpt}
            self.rebuild()
            self.ann_changed.emit()
        else:
            # 开始放置新的关键点
            self.selected_index = None
            n_kpts = len(self.ds.meta.keypoints)
            if n_kpts == 0:
                n_kpts = 17
            self._drawing = {"mode": "new_pose", "points": [p], "total": n_kpts}
            self.rebuild()
            self._preview_kpts()

    def _preview_kpts(self):
        self._clear_preview()
        color = self._selected_color(self._current_class)
        for pt in self._drawing["points"]:
            self._preview_items.append(self._scene.addEllipse(pt.x() - 4, pt.y() - 4, 8, 8, QPen(color), QBrush(color)))

    def mouseMoveEvent(self, ev):
        if self._drawing is None:
            return
        p = self.mapToScene(ev.position().toPoint())
        if self._drawing["mode"] == "move_kpt":
            ann = self.anns[self._drawing["index"]]
            k = self._drawing["kpt"]
            x, y = self.pix_to_norm(p)
            ann["keypoints"][k][0] = x
            ann["keypoints"][k][1] = y
            ann["keypoints"][k][2] = max(ann["keypoints"][k][2], 2)
            self.rebuild()

    def mouseReleaseEvent(self, ev):
        if ev.button() != Qt.LeftButton or self._drawing is None:
            return
        p = self.mapToScene(ev.position().toPoint())
        if self._drawing["mode"] == "move_kpt":
            ann = self.anns[self._drawing["index"]]
            k = self._drawing["kpt"]
            x, y = self.pix_to_norm(p)
            ann["keypoints"][k][0] = x
            ann["keypoints"][k][1] = y
            # 重新计算 bbox
            kps = ann["keypoints"]
            xs = [k0[0] for k0 in kps if k0[2] > 0]
            ys = [k0[1] for k0 in kps if k0[2] > 0]
            if xs:
                ann["bbox"] = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
                               max(0.02, max(xs) - min(xs) + 0.04), max(0.02, max(ys) - min(ys) + 0.04)]
            self._drawing = None
            self.rebuild()
            self.ann_changed.emit()
        elif self._drawing["mode"] == "new_pose":
            self._drawing["points"].append(p)
            if len(self._drawing["points"]) >= self._drawing["total"]:
                pts = [self.pix_to_norm(pt) for pt in self._drawing["points"]]
                xs = [x for x, y in pts]
                ys = [y for x, y in pts]
                ann = {"class_id": self._current_class,
                       "bbox": [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
                                max(0.02, max(xs) - min(xs) + 0.04), max(0.02, max(ys) - min(ys) + 0.04)],
                       "keypoints": [[x, y, 2] for x, y in pts]}
                self._drawing = None
                self._clear_preview()
                self._commit_ann(ann)
            else:
                self._preview_kpts()


# ================================================================ 定向边界框
class ObbCanvas(BaseCanvas):
    def rebuild(self):
        self._reset_ann_items()
        for i, ann in enumerate(self.anns):
            color = self._selected_color(ann["class_id"])
            cx, cy = self.norm_to_pix(ann["bbox"][0], ann["bbox"][1])
            w = ann["bbox"][2] * self._img_w
            h = ann["bbox"][3] * self._img_h
            ang = math.radians(ann.get("angle", 0.0))
            rect = QRectF(cx.x() - w / 2, cy.y() - h / 2, w, h)
            item = self._scene.addRect(rect, QPen(color, 3 if i == self.selected_index else 2))
            item.setData(0, i)
            item.setRotation(-ann.get("angle", 0.0))
            item.setTransformOriginPoint(cx)
            self._ann_items.append(item)
            # 旋转手柄
            hx = cx.x() + math.cos(ang) * w / 2
            hy = cy.y() - math.sin(ang) * w / 2
            handle = self._scene.addEllipse(hx - 4, hy - 4, 8, 8, QPen(color), QBrush(color))
            handle.setData(0, i)
            self._ann_items.append(handle)

    def _ann_center(self, ann) -> QPointF:
        return self.norm_to_pix(ann["bbox"][0], ann["bbox"][1])

    def _handle_pos(self, ann) -> QPointF:
        cx, cy = self._ann_center(ann).x(), self._ann_center(ann).y()
        w = ann["bbox"][2] * self._img_w
        ang = math.radians(ann.get("angle", 0.0))
        return QPointF(cx + math.cos(ang) * w / 2, cy - math.sin(ang) * w / 2)

    def _hit_test(self, p: QPointF):
        for i in range(len(self.anns) - 1, -1, -1):
            ann = self.anns[i]
            c = self._ann_center(ann)
            w = ann["bbox"][2] * self._img_w
            h = ann["bbox"][3] * self._img_h
            if w == 0 or h == 0:
                continue
            dx, dy = p.x() - c.x(), p.y() - c.y()
            ang = math.radians(ann.get("angle", 0.0))
            lx = dx * math.cos(ang) - dy * math.sin(ang)
            ly = dx * math.sin(ang) + dy * math.cos(ang)
            if abs(lx) <= w / 2 + self.point_threshold and abs(ly) <= h / 2 + self.point_threshold:
                if self._dist(p, self._handle_pos(ann)) <= self.point_threshold * 1.5:
                    return i, "rotate"
                return i, "move"
        return None, None

    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            return super().mousePressEvent(ev)
        p = self.mapToScene(ev.position().toPoint())
        idx, mode = self._hit_test(p)
        if idx is not None:
            self.snapshot()
            self.selected_index = idx
            self._drawing = {"mode": mode, "index": idx, "start": p}
            self.rebuild()
            self.ann_changed.emit()
        else:
            self.selected_index = None
            self._drawing = {"mode": "new", "start": p}
            self.rebuild()

    def mouseMoveEvent(self, ev):
        if self._drawing is None:
            return
        p = self.mapToScene(ev.position().toPoint())
        mode = self._drawing["mode"]
        if mode == "new":
            x0, y0 = self._drawing["start"].x(), self._drawing["start"].y()
            rect = QRectF(x0, y0, p.x() - x0, p.y() - y0).normalized()
            self._clear_preview()
            self._preview_items.append(self._scene.addRect(rect,
                QPen(QColor(self._selected_color(self._current_class)), 2, Qt.DashLine)))
        elif mode == "move":
            dx = (p.x() - self._drawing["start"].x()) / self._img_w
            dy = (p.y() - self._drawing["start"].y()) / self._img_h
            ann = self.anns[self._drawing["index"]]
            ann["bbox"][0] = min(1.0, max(0.0, ann["bbox"][0] + dx))
            ann["bbox"][1] = min(1.0, max(0.0, ann["bbox"][1] + dy))
            self.rebuild()
        elif mode == "rotate":
            ann = self.anns[self._drawing["index"]]
            c = self._ann_center(ann)
            ang = math.degrees(math.atan2(c.y() - p.y(), p.x() - c.x()))
            ann["angle"] = ang % 180
            self.rebuild()

    def mouseReleaseEvent(self, ev):
        if ev.button() != Qt.LeftButton or self._drawing is None:
            return
        p = self.mapToScene(ev.position().toPoint())
        mode = self._drawing["mode"]
        self._clear_preview()
        if mode == "new":
            x0, y0 = self._drawing["start"].x(), self._drawing["start"].y()
            rect = QRectF(x0, y0, p.x() - x0, p.y() - y0).normalized()
            if rect.width() > 6 and rect.height() > 6 and self._img_w > 0:
                ann = {"class_id": self._current_class,
                       "bbox": [rect.center().x() / self._img_w, rect.center().y() / self._img_h,
                                rect.width() / self._img_w, rect.height() / self._img_h],
                       "angle": 0.0}
                self._commit_ann(ann)
        elif mode in ("move", "rotate"):
            self.rebuild()
            self.ann_changed.emit()
        self._drawing = None


# ================================================================ 图像分类
class ClassifyCanvas(BaseCanvas):
    def rebuild(self):
        self._reset_ann_items()
        cid = self.ds.classify_label(self.editor.current_image_name)
        if cid is not None:
            color = self._selected_color(cid)
            name = self.ds.meta.classes[cid].name
            text = self._scene.addText(f"✔ {name}")
            text.setDefaultTextColor(color)
            text.setPos(10, 10)
            self._ann_items.append(text)

    def mousePressEvent(self, ev):
        pass  # 分类任务无绘制操作


# ================================================================ 编辑器外壳
class AnnotationEditor(QWidget):
    """标注编辑器：左侧图像列表、中央画布、右侧标注控制"""
    closed = Signal()

    def __init__(self, ctx, ds, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.ds = ds
        self.current_image_name = None
        self._modified = False
        self._build_ui()
        self.refresh_image_list()
        if self.image_list.count():
            self.image_list.setCurrentRow(0)

    # ---------------- UI ----------------
    def _build_ui(self):
        self.setWindowTitle(f"标注 - {self.ds.name}（{task_label(self.ds.task)}）")
        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.title_label = QLabel(f"📝 {self.ds.name} · {task_label(self.ds.task)}")
        self.title_label.setObjectName("title")
        top.addWidget(self.title_label)
        top.addStretch(1)
        self.ai_badge = QLabel("")
        self.ai_badge.setObjectName("warning")
        top.addWidget(self.ai_badge)
        root.addLayout(top)

        split = QSplitter(Qt.Horizontal)
        # 左：图像列表
        self.image_list = QListWidget()
        self.image_list.setIconSize(self.image_list.iconSize())
        self.image_list.setMaximumWidth(260)
        self.image_list.currentRowChanged.connect(self._on_image_selected)
        split.addWidget(self.image_list)

        # 中：画布
        self.canvas = self._make_canvas()
        split.addWidget(self.canvas)

        # 右：控制面板
        panel = QWidget()
        panel.setMaximumWidth(300)
        play = QVBoxLayout(panel)
        play.addWidget(QLabel("类别"))
        self.class_combo = QComboBox()
        for c in self.ds.meta.classes:
            self.class_combo.addItem(f"{c.name}", c.id)
        self.class_combo.currentIndexChanged.connect(self._on_class_changed)
        play.addWidget(self.class_combo)

        play.addWidget(QLabel("标注列表"))
        self.ann_table = QTableWidget(0, 2)
        self.ann_table.setHorizontalHeaderLabels(["类别", "详情"])
        self.ann_table.horizontalHeader().setStretchLastSection(True)
        self.ann_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ann_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        play.addWidget(self.ann_table, 1)

        self.modified_label = QLabel("")
        self.modified_label.setObjectName("warning")
        play.addWidget(self.modified_label)

        btn_save = QPushButton("💾 保存标注 (Ctrl+S)")
        btn_save.setObjectName("primary")
        btn_save.clicked.connect(self.save_current)
        play.addWidget(btn_save)
        row = QHBoxLayout()
        btn_undo = QPushButton("撤销")
        btn_undo.clicked.connect(self.canvas.undo)
        btn_redo = QPushButton("重做")
        btn_redo.clicked.connect(self.canvas.redo)
        btn_del = QPushButton("删除")
        btn_del.setObjectName("danger")
        btn_del.clicked.connect(self._delete_annotation)
        row.addWidget(btn_undo)
        row.addWidget(btn_redo)
        row.addWidget(btn_del)
        play.addLayout(row)

        nav = QHBoxLayout()
        self.btn_prev = QPushButton("◀ 上一张")
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next = QPushButton("下一张 ▶")
        self.btn_next.clicked.connect(self.next_image)
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        play.addLayout(nav)

        self.btn_add = QPushButton("➕ 添加图像...")
        self.btn_add.clicked.connect(self._add_images)
        play.addWidget(self.btn_add)

        self.btn_ai = QPushButton("🤖 AI 标注...")
        self.btn_ai.clicked.connect(self._open_ai_annotate)
        play.addWidget(self.btn_ai)

        self.progress_label = QLabel("")
        self.progress_label.setObjectName("dim")
        play.addWidget(self.progress_label)

        play.addStretch(1)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        play.addWidget(btn_close)
        split.addWidget(panel)
        split.setSizes([240, 900, 280])
        root.addWidget(split, 1)
        self.resize(1500, 950)

    def _make_canvas(self):
        task = self.ds.task
        if task == "detect":
            canvas = DetectCanvas(self.ds, self)
        elif task == "segment":
            canvas = SegmentCanvas(self.ds, self)
        elif task == "pose":
            canvas = PoseCanvas(self.ds, self)
        elif task == "obb":
            canvas = ObbCanvas(self.ds, self)
        else:
            canvas = ClassifyCanvas(self.ds, self)
        canvas.ann_changed.connect(self._on_canvas_changed)
        canvas._current_class = self.ds.meta.classes[0].id if self.ds.meta.classes else 0
        return canvas

    # ---------------- 图像列表（缩略图分批懒加载，大数据集不卡界面） ----------------
    def refresh_image_list(self, select: str = None):
        self.image_list.blockSignals(True)
        self.image_list.clear()
        self._icon_gen = getattr(self, "_icon_gen", 0) + 1
        gen = self._icon_gen
        self._icon_queue = []
        self._image_paths = self.ds.image_files()
        for img in self._image_paths:
            name = self.ds.image_display_name(img)
            item = QListWidgetItem(name)
            suffix = " 🤖" if self.ds.is_ai(name) else ""
            item.setText(f"{name}{suffix}")
            item.setToolTip(name)
            self.image_list.addItem(item)
            self._icon_queue.append((img, item))
            if select and name == select:
                self.image_list.setCurrentRow(self.image_list.count() - 1)
        self.image_list.blockSignals(False)
        self.progress_label.setText(f"共 {len(self._image_paths)} 张图像")
        self._load_icons_stepwise(gen)

    def _load_icons_stepwise(self, gen: int):
        """每批加载 40 张缩略图，用定时器分片，保持界面响应"""
        if gen != getattr(self, "_icon_gen", 0):
            return  # 列表已刷新，丢弃旧队列
        from PySide6.QtCore import QTimer
        from PySide6.QtGui import QIcon
        batch = self._icon_queue[:40]
        del self._icon_queue[:40]
        for img, item in batch:
            thumb = self.ctx.cache.get(img)
            if thumb:
                item.setIcon(QIcon(str(thumb)))
        if self._icon_queue:
            QTimer.singleShot(50, lambda: self._load_icons_stepwise(gen))

    # ---------------- 图像切换 ----------------
    def _on_image_selected(self, row: int):
        if row < 0 or row >= len(getattr(self, "_image_paths", [])):
            return
        img = self._image_paths[row]
        img_name = self.ds.image_display_name(img)
        self.current_image_name = img_name
        self.canvas.set_image(img)
        anns = self.ds.read_annotations(img_name)
        self.canvas.load_annotations(anns)
        self._modified = False
        self.modified_label.setText("")
        self.ai_badge.setText("🤖 AI 标注" if self.ds.is_ai(img_name) else "")
        if self.ds.task == "classify":
            cid = self.ds.classify_label(img_name)
            if cid is not None:
                idx = self.class_combo.findData(cid)
                if idx >= 0:
                    self.class_combo.blockSignals(True)
                    self.class_combo.setCurrentIndex(idx)
                    self.class_combo.blockSignals(False)
        self._refresh_ann_table()

    def _refresh_ann_table(self):
        self.ann_table.setRowCount(0)
        for ann in self.canvas.anns:
            row = self.ann_table.rowCount()
            self.ann_table.insertRow(row)
            cls = self.ds.class_by_id(ann["class_id"])
            self.ann_table.setItem(row, 0, QTableWidgetItem(cls.name if cls else str(ann["class_id"])))
            if self.ds.task == "detect":
                detail = f"bbox {', '.join(f'{v:.2f}' for v in ann['bbox'])}"
            elif self.ds.task == "obb":
                detail = f"bbox ... 角度 {ann.get('angle', 0):.1f}°"
            elif self.ds.task == "pose":
                visible = sum(1 for _, _, v in ann["keypoints"] if v == 2)
                detail = f"{len(ann['keypoints'])} 关键点（{visible} 可见）"
            elif self.ds.task == "segment":
                detail = f"{len(ann['points'])} 顶点"
            else:
                detail = ""
            self.ann_table.setItem(row, 1, QTableWidgetItem(detail))

    def _on_class_changed(self):
        self.canvas.set_class_id(self.class_combo.currentData())
        if self.ds.task == "classify" and self.current_image_name:
            cls = self.ds.class_by_id(self.class_combo.currentData())
            if cls:
                self.ds.set_classify_label(self.current_image_name, cls.name)
                self._modified = True
                self.modified_label.setText("已修改，请保存")
                self._refresh_ann_table()

    def _on_canvas_changed(self):
        self._modified = True
        self.modified_label.setText("已修改，请保存")
        self._refresh_ann_table()

    def _delete_annotation(self):
        self.canvas.delete_selected()

    # ---------------- 保存 ----------------
    def save_current(self):
        if self.current_image_name is None:
            return
        if self.ds.task == "classify":
            # 分类任务：标签即目录，已在切换时应用
            pass
        else:
            self.ds.write_annotations(self.current_image_name, self.canvas.to_annotations())
        if self._modified:
            self.ds.unmark_ai(self.current_image_name)
        self._modified = False
        self.modified_label.setText("✔ 已保存")
        self.ai_badge.setText("")
        self.ctx.activity.log("保存标注", self.ds.name, self.current_image_name,
                              actor=self.ctx.auth.current_username)
        self.refresh_image_list(select=self.current_image_name)

    def prev_image(self):
        row = self.image_list.currentRow()
        if row > 0:
            self._save_before_switch()
            self.image_list.setCurrentRow(row - 1)

    def next_image(self):
        row = self.image_list.currentRow()
        if row < self.image_list.count() - 1:
            self._save_before_switch()
            self.image_list.setCurrentRow(row + 1)

    def _save_before_switch(self):
        if self._modified:
            self.save_current()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Tab:
            self.next_image()
        elif ev.key() == Qt.Key_S and ev.modifiers() & Qt.ControlModifier:
            self.save_current()
        elif ev.key() == Qt.Key_Z and ev.modifiers() & Qt.ControlModifier:
            self.canvas.undo()
        elif ev.key() == Qt.Key_Y and ev.modifiers() & Qt.ControlModifier:
            self.canvas.redo()
        elif ev.key() == Qt.Key_Delete and self.canvas.selected_index is not None:
            self.canvas.delete_selected()
        else:
            super().keyPressEvent(ev)

    # ---------------- 添加图像（后台线程） ----------------
    def _add_images(self):
        from PySide6.QtWidgets import QFileDialog, QProgressDialog
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图像", "",
            "图像 (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not files:
            return
        from pathlib import Path
        from gui.workers import FunctionWorker
        prog = QProgressDialog("正在添加图像 ...", "", 0, 100, self)
        prog.setCancelButton(None)
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(300)
        worker = FunctionWorker(
            lambda cb: self.ds.add_images([Path(f) for f in files], progress_cb=cb))
        worker.progress.connect(lambda d, t: (prog.setMaximum(max(t, 1)), prog.setValue(d)))

        def done(added):
            prog.close()
            self.ctx.activity.log("添加图像", self.ds.name, f"{added} 张",
                                  actor=self.ctx.auth.current_username)
            self.refresh_image_list(select=self.current_image_name)

        worker.succeeded.connect(done)
        worker.failed.connect(lambda m: (prog.close(), msg_error(self, "添加失败", m)))
        self._add_worker = worker
        worker.start()

    # ---------------- AI 标注 ----------------
    def _open_ai_annotate(self):
        from gui.widgets.ai_auto_annotate import AIAnnotateDialog
        dlg = AIAnnotateDialog(self.ctx, self.ds, self)
        if dlg.exec():
            self.refresh_image_list(select=self.current_image_name)
            self._on_image_selected(self.image_list.currentRow())
