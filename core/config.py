# -*- coding: utf-8 -*-
"""配置管理系统：数据类 + YAML 持久化（配置驱动，所有参数可改 config/config.yaml）"""
import threading
from dataclasses import dataclass, field, fields, is_dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional

import yaml


@dataclass
class AppConfig:
    name: str = "YOLO Platform V2.0"
    version: str = "2.0.0"
    debug: bool = False
    log_level: str = "INFO"
    language: str = "zh_CN"


@dataclass
class DataConfig:
    supported_formats: List[str] = field(default_factory=lambda: ["coco", "yolo", "pascal_voc"])
    default_split_ratio: Dict[str, float] = field(default_factory=lambda: {"train": 0.7, "val": 0.2, "test": 0.1})
    max_workers: int = 4


@dataclass
class AugConfig:
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0


@dataclass
class ModelConfig:
    default_model: str = "yolo11n"
    device: str = "auto"
    mixed_precision: bool = True
    supported_series: List[str] = field(default_factory=lambda: ["yolov8", "yolo11", "yolo12"])


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    learning_rate: float = 0.01
    final_lr_factor: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    optimizer: str = "auto"
    scheduler: str = "auto"
    early_stopping: bool = True
    patience: int = 50
    workers: int = 8


@dataclass
class UIConfig:
    theme: str = "dark"
    font_family: str = "Microsoft YaHei"
    font_size: int = 10
    window_width: int = 1400
    window_height: int = 900


@dataclass
class PathConfig:
    data_root: str = "workspace/datasets"
    models_root: str = "workspace/models"
    outputs_root: str = "workspace/outputs"
    cache_root: str = "workspace/cache"
    logs_root: str = "logs"
    config_root: str = "config"


@dataclass
class SecurityConfig:
    session_timeout_minutes: int = 30
    api_host: str = "127.0.0.1"
    api_port: int = 8000


@dataclass
class Config:
    app: AppConfig = field(default_factory=AppConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugConfig = field(default_factory=AugConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)


def _from_dict(cls, data: dict):
    kwargs = {}
    for f in fields(cls):
        if f.name in data:
            val = data[f.name]
            if isinstance(val, dict) and is_dataclass(f.type):
                kwargs[f.name] = _from_dict(f.type, val)
            else:
                kwargs[f.name] = val
    return cls(**kwargs)


class ConfigManager:
    """配置管理器（单例）。首次运行自动生成默认 config/config.yaml。"""
    _instance: Optional["ConfigManager"] = None
    _lock = threading.Lock()

    def __init__(self, project_root: Path = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
        self.config_path = self.project_root / "config" / "config.yaml"
        self.config = Config()

    @classmethod
    def get_instance(cls, project_root: Path = None) -> "ConfigManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(project_root)
            return cls._instance

    def load(self) -> Config:
        if self.config_path.exists():
            try:
                raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
                self.config = _from_dict(Config, raw)
            except Exception:
                self.config = Config()
                self.save()
        else:
            self.save()
        return self.config

    def save(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            yaml.safe_dump(asdict(self.config), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def to_dict(self) -> dict:
        return asdict(self.config)

    def update_section(self, section: str, data: dict):
        """更新某一配置节，如 update_section('training', {'epochs': 200})"""
        sec = getattr(self.config, section, None)
        if sec is None:
            raise ValueError(f"未知配置节: {section}")
        for k, v in data.items():
            if hasattr(sec, k):
                setattr(sec, k, v)
        self.save()

    def resolve_path(self, rel: str) -> Path:
        """相对项目根的路径，自动创建目录"""
        p = (self.project_root / rel).resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    # 常用路径
    @property
    def datasets_root(self) -> Path:
        return self.resolve_path(self.config.paths.data_root)

    @property
    def models_root(self) -> Path:
        return self.resolve_path(self.config.paths.models_root)

    @property
    def outputs_root(self) -> Path:
        return self.resolve_path(self.config.paths.outputs_root)

    @property
    def cache_root(self) -> Path:
        return self.resolve_path(self.config.paths.cache_root)

    @property
    def logs_root(self) -> Path:
        return self.resolve_path(self.config.paths.logs_root)
