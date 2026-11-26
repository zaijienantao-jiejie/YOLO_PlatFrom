# -*- coding: utf-8 -*-
"""环境与依赖检测（供系统诊断使用），并统一设备选择逻辑"""
import importlib
import platform
import sys
from typing import List, Optional

MODULES = {
    "python": "platform",
    "PySide6": "PySide6",
    "OpenCV": "cv2",
    "Pillow": "PIL",
    "NumPy": "numpy",
    "Pandas": "pandas",
    "PyYAML": "yaml",
    "PyTorch": "torch",
    "Ultralytics": "ultralytics",
    "FastAPI": "fastapi",
    "Uvicorn": "uvicorn",
    "Werkzeug": "werkzeug",
    "Matplotlib": "matplotlib",
    "Seaborn": "seaborn",
    "ONNX": "onnx",
    "psutil": "psutil",
}


def _module_version(mod):
    for attr in ("__version__", "version"):
        v = getattr(mod, attr, None)
        if isinstance(v, str):
            return v
    return None


def check_dependencies() -> dict:
    """返回 {显示名: 版本字符串或 None}"""
    out = {}
    for name, module_name in MODULES.items():
        if name == "python":
            out[name] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            continue
        try:
            out[name] = _module_version(importlib.import_module(module_name)) or "已安装"
        except ImportError:
            out[name] = None
    return out


def is_full_install() -> bool:
    deps = check_dependencies()
    return all(deps.get(k) for k in ("PyTorch", "Ultralytics", "FastAPI", "PySide6"))


def is_minimal_install() -> bool:
    return not any(check_dependencies().get(k) for k in ("PyTorch", "Ultralytics"))


def resolve_device(device: str) -> str:
    """device: 'auto' → 有 CUDA 用 0 号卡，否则 CPU；其余原样返回"""
    if device and device not in ("auto", ""):
        return device
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def torch_device_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "available": True,
                "device": "cuda:0",
                "gpu": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
            }
        return {"available": False, "device": "cpu", "gpu": None, "cuda_version": None, "gpu_count": 0}
    except ImportError:
        return {"available": False, "device": "cpu", "gpu": None, "cuda_version": None, "gpu_count": 0}


def system_info() -> dict:
    import shutil
    import os
    cpu = platform.processor() or "Unknown"
    ram_gb = None
    disk_free_gb = None
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / 1024 ** 3, 1)
        disk_free_gb = round(psutil.disk_usage(str(os.getcwd())).free / 1024 ** 3, 1)
        cpu = platform.processor() or "Unknown"
        cores = psutil.cpu_count(logical=True)
    except ImportError:
        cores = os.cpu_count()
    return {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "cpu": cpu,
        "cpu_cores": cores,
        "ram_gb": ram_gb,
        "disk_free_gb": disk_free_gb,
    }
