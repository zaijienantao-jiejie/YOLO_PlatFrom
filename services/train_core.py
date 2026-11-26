# -*- coding: utf-8 -*-
"""训练核心逻辑（无 GUI 依赖）：供命令行 train_cli.py 与 QThread 封装共用

AutoDL 等云服务器上不需要安装 PySide6 即可使用。
"""
from pathlib import Path

from core import logger
from core.dependencies import resolve_device

TASK_SUFFIX = {"detect": "", "segment": "-seg", "pose": "-pose", "obb": "-obb", "classify": "-cls"}


def ensure_task_suffix(model_name: str, task: str) -> str:
    """yolo11n + pose → yolo11n-pose"""
    if task == "detect":
        return model_name
    suffix = TASK_SUFFIX.get(task, "")
    if suffix and not model_name.endswith(suffix):
        return model_name + suffix
    return model_name


def run_training(data_yaml: str, model_name: str, params: dict, project: str,
                 device: str = "auto", resume: bool = False, pretrained_dir=None,
                 on_epoch=None, on_status=None, stop_flag: dict = None):
    """训练核心逻辑。

    on_epoch(dict) / on_status(str) 为可选回调；stop_flag 为共享 dict，
    外部置 stop_flag["stop"] = True 可在当前 epoch 结束后停止。
    返回 (best 模型路径, 结果目录)；被停止时返回 (None, None)。
    """
    from ultralytics import YOLO
    from services.model_utils import resolve_model_file

    model_path = resolve_model_file(model_name, pretrained_dir)
    if on_status:
        if model_path != model_name:
            on_status(f"使用本地模型 {model_path}")
        else:
            on_status(f"正在下载/加载模型 {model_name} ...")
    model = YOLO(model_path)

    def _epoch_cb(trainer):
        if stop_flag is not None and stop_flag.get("stop"):
            trainer.stop = True
            return
        if on_epoch is None:
            return
        m = trainer.metrics or {}
        out = {"epoch": int(getattr(trainer, "epoch", 0)) + 1}
        for key in ("train/box_loss", "train/seg_loss", "train/cls_loss", "train/dfl_loss",
                    "train/pose_loss", "train/obb_loss"):
            if key in m:
                out[key] = float(m[key])
        mapping = {
            "metrics/precision(B)": "precision", "metrics/recall(B)": "recall",
            "metrics/mAP50(B)": "mAP50", "metrics/mAP50-95(B)": "mAP50-95",
            "metrics/accuracy_top1": "top1", "metrics/accuracy_top5": "top5",
            "metrics/mAP50(P)": "mAP50-pose", "metrics/mAP50-95(P)": "mAP50-95-pose",
        }
        for src, dst in mapping.items():
            if src in m:
                out[dst] = float(m[src])
        lr = getattr(trainer, "lr", None)
        if lr is not None:
            try:
                import numpy as np
                out["lr"] = float(np.ravel(lr)[0])
            except Exception:
                pass
        on_epoch(out)

    model.add_callback("on_train_epoch_end", _epoch_cb)
    kwargs = {
        "data": data_yaml,
        "project": project,
        "name": f"train_{model_name}",
        "device": resolve_device(device),
        "resume": resume,
        "exist_ok": True,
        "plots": True,
        "verbose": False,
        "val": True,
    }
    kwargs.update(params)
    if on_status:
        on_status(f"开始训练（数据: {data_yaml}）...")
    model.train(**kwargs)
    if stop_flag is not None and stop_flag.get("stop"):
        if on_status:
            on_status("训练已停止（可断点续训）")
        return None, None
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if on_status:
        on_status("训练完成")
    return str(best), str(save_dir)
