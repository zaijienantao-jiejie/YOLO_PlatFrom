# -*- coding: utf-8 -*-
"""应用上下文：把全局服务注入各 GUI 组件"""
from dataclasses import dataclass

from auth.manager import AuthManager
from core.activity_log import ActivityLogger
from core.cache_manager import ThumbnailCache
from core.config import ConfigManager
from data.dataset_manager import DatasetManager


@dataclass
class AppContext:
    cm: ConfigManager
    auth: AuthManager
    dm: DatasetManager
    activity: ActivityLogger
    cache: ThumbnailCache
