# -*- coding: utf-8 -*-
"""平台自定义异常体系"""


class PlatformError(Exception):
    """平台基础异常"""


class ConfigError(PlatformError):
    """配置错误"""


class DatasetError(PlatformError):
    """数据集错误"""


class ValidationError(PlatformError):
    """数据验证错误"""


class AnnotationError(PlatformError):
    """标注错误"""


class TrainingError(PlatformError):
    """训练错误"""


class EvaluationError(PlatformError):
    """评估错误"""


class ExportError(PlatformError):
    """模型导出错误"""


class InferenceError(PlatformError):
    """推理错误"""


class AuthError(PlatformError):
    """认证错误"""
