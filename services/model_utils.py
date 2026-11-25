# -*- coding: utf-8 -*-
"""模型文件解析：统一查找本地预训练权重，避免重复下载

ultralytics 官方查找顺序（utils/downloads.py）：
  1. 当前工作目录下的同名文件
  2. ultralytics 全局权重目录（SETTINGS["weights_dir"]，如 %APPDATA%/Ultralytics/weights）
  3. 都没有 → 从 https://github.com/ultralytics/assets/releases/download/... 下载到当前目录
"""
from pathlib import Path
from typing import Optional


def resolve_model_file(model_name: str, pretrained_dir=None) -> str:
    """解析模型文件位置，返回可直接传给 YOLO() 的路径。

    查找顺序：
      1. model_name 本身已存在（完整路径或当前目录文件）
      2. <pretrained_dir>/<文件名>（默认 workspace/models/pretrained/）
      3. ultralytics 全局权重目录
      4. 都找不到 → 原样返回，由 ultralytics 自动从 GitHub 下载
    """
    name = str(model_name).strip().strip("'\"")
    p = Path(name)
    if p.exists():
        return str(p)
    if p.is_absolute() or p.parent != Path("."):
        return name  # 带路径但不存在：交给 ultralytics 报错或下载

    candidates = []
    if pretrained_dir:
        candidates.append(Path(pretrained_dir) / p.name)
    candidates.append(Path.cwd() / p.name)
    try:
        from ultralytics.utils import SETTINGS
        candidates.append(Path(SETTINGS["weights_dir"]) / p.name)
    except ImportError:
        pass
    for c in candidates:
        if c.exists():
            return str(c)
    return name
