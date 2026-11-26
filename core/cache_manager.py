# -*- coding: utf-8 -*-
"""缓存管理：图像缩略图 LRU 缓存"""
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from core import logger


class ThumbnailCache:
    """LRU 缩略图缓存：GUI 列表快速显示图像"""

    def __init__(self, cache_dir: Path, max_entries: int = 2000, size: tuple = (256, 256)):
        self.cache_dir = Path(cache_dir) / "thumbnails"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self.size = size
        self._order: OrderedDict[str, None] = OrderedDict()

    def _thumb_path(self, image_path, mtime: float) -> Path:
        name = f"{abs(hash(str(image_path))):x}_{int(mtime):x}.jpg"
        return self.cache_dir / name

    def get(self, image_path) -> Optional[Path]:
        """返回缩略图文件路径；图像未变化时命中缓存"""
        p = Path(image_path)
        try:
            mtime = p.stat().st_mtime
        except OSError:
            return None
        thumb = self._thumb_path(p, mtime)
        if thumb.exists():
            self._order.pop(str(p), None)
            self._order[str(p)] = None
            return thumb
        try:
            from PIL import Image
            img = Image.open(p)
            img.thumbnail(self.size)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(thumb, "JPEG", quality=85)
        except Exception as e:
            logger.get_logger().warning(f"缩略图生成失败 {p}: {e}")
            return None
        self._order[str(p)] = None
        self._evict()
        return thumb

    def _evict(self):
        while len(self._order) > self.max_entries:
            oldest, _ = self._order.popitem(last=False)
            # 尽量删除对应缩略图文件
            for t in self.cache_dir.glob(f"{abs(hash(oldest)):x}_*.jpg"):
                try:
                    t.unlink()
                except OSError:
                    pass

    def invalidate(self, image_path):
        p = str(image_path)
        self._order.pop(p, None)
        for t in self.cache_dir.glob(f"{abs(hash(p)):x}_*.jpg"):
            try:
                t.unlink()
            except OSError:
                pass

    def clear(self):
        self._order.clear()
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def total_size_mb(self) -> float:
        total = sum(f.stat().st_size for f in self.cache_dir.glob("*") if f.is_file())
        return total / 1024 / 1024
