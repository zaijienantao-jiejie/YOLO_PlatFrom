# -*- coding: utf-8 -*-
"""核心模块：配置、日志、主题、版本、缓存、依赖检测"""
from core.version import APP_NAME, APP_VERSION
from core.exceptions import (
    PlatformError, ConfigError, DatasetError, ValidationError, AnnotationError,
    TrainingError, EvaluationError, ExportError, InferenceError, AuthError,
)
