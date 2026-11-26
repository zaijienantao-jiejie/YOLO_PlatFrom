# -*- coding: utf-8 -*-
"""命令行训练入口 —— 用于 AutoDL 等无图形界面的云 GPU 环境。

示例：
  # 查看平台里有哪些数据集
  python train_cli.py --list-datasets

  # 训练平台数据集（自动生成 data.yaml、自动划分）
  python train_cli.py --dataset 药材分类导入测试 --model yolo11n-cls \
      --epochs 100 --batch 64 --imgsz 224 --device 0

  # 断点续训
  python train_cli.py --dataset 药材分类导入测试 --model yolo11n-cls --resume

  # 直接指定 data.yaml
  python train_cli.py --data path/to/data.yaml --model yolo11n --epochs 50 --device 0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import logger as logger_mod
from core.config import ConfigManager
from core.dependencies import torch_device_info
from services.train_core import run_training, ensure_task_suffix


def main():
    ap = argparse.ArgumentParser(description="YOLO Platform 命令行训练")
    ap.add_argument("--dataset", help="平台数据集名称（workspace/datasets 下）")
    ap.add_argument("--data", help="或直接指定 data.yaml 路径")
    ap.add_argument("--model", default=None, help="模型名，如 yolo11n / yolo11n-cls / yolo11s（--list-datasets 时不需要）")
    ap.add_argument("--task", default=None,
                    help="任务类型 detect/segment/pose/obb/classify（--data 时按模型名推断不了才需要）")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--device", default="auto", help="auto / 0 / cpu")
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true", help="从 last.pt 断点续训")
    ap.add_argument("--amp", action="store_true", help="启用混合精度训练")
    ap.add_argument("--project", default=None, help="结果输出目录（默认 workspace/models）")
    ap.add_argument("--list-datasets", action="store_true", help="列出平台数据集后退出")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent
    cm = ConfigManager(project_root)
    cm.load()
    logger_mod.setup_logging(cm.logs_root, cm.config.app.log_level)

    from data.dataset_manager import DatasetManager
    dm = DatasetManager(cm.datasets_root)

    if args.list_datasets:
        ds_list = dm.list()
        if not ds_list:
            print("（没有数据集。请把本地 workspace/datasets/ 下的数据集文件夹上传到对应位置）")
            return
        print("平台数据集列表：")
        for ds in ds_list:
            print(f"  {ds.name}\t{ds.task}\t{len(ds.image_files())} 张图像")
        return

    # 解析数据源
    if not args.model:
        ap.error("请指定 --model（或使用 --list-datasets 查看数据集）")
    if args.dataset:
        ds = dm.get(args.dataset)
        if ds.task != "classify":
            counts = ds.generate_splits()
            print(f"数据划分: {counts}")
        data_yaml = str(ds.build_yaml())
        task = ds.task
    elif args.data:
        data_yaml = args.data
        task = args.task
        if not task:
            # 从模型名推断任务
            for t, suffix in (("classify", "-cls"), ("segment", "-seg"), ("pose", "-pose"), ("obb", "-obb")):
                if args.model.endswith(suffix):
                    task = t
                    break
            else:
                task = "detect"
    else:
        ap.error("请指定 --dataset 或 --data")

    model_name = ensure_task_suffix(args.model, task)
    params = {
        "epochs": args.epochs, "batch": args.batch, "imgsz": args.imgsz,
        "lr0": args.lr0, "patience": args.patience, "workers": args.workers,
    }
    if args.amp:
        params["amp"] = True
    project = args.project or str(cm.models_root)

    info = torch_device_info()
    if info["available"]:
        print(f"设备: GPU {info['gpu']} (CUDA {info['cuda_version']})")
    else:
        print("设备: CPU（未检测到 CUDA，训练会很慢）")
    print(f"任务: {task} | 数据: {data_yaml}")
    print(f"模型: {model_name} | 参数: {params} | 设备参数: {args.device}\n")

    def on_status(msg):
        print(f"[状态] {msg}", flush=True)

    def on_epoch(out):
        parts = [f"epoch {out.get('epoch')}"]
        for k in ("train/box_loss", "train/cls_loss", "train/seg_loss", "train/pose_loss", "train/dfl_loss"):
            if k in out:
                parts.append(f"{k.split('/')[-1]} {out[k]:.4f}")
        for k in ("mAP50", "mAP50-95", "top1", "top5", "precision", "recall"):
            if k in out:
                parts.append(f"{k} {out[k]:.4f}")
        if "lr" in out:
            parts.append(f"lr {out['lr']:.5f}")
        print("  " + " | ".join(parts), flush=True)

    best, save_dir = run_training(
        data_yaml, model_name, params, project,
        device=args.device, resume=args.resume,
        pretrained_dir=str(cm.models_root / "pretrained"),
        on_epoch=on_epoch, on_status=on_status)

    if best:
        print(f"\n训练完成！")
        print(f"  最佳模型: {best}")
        print(f"  结果目录: {save_dir}")
        print("把 best.pt 下载回本地后，可在平台“评估/部署”页使用；也可在 AutoDL 上继续训练/导出。")
    else:
        print("\n训练已停止（可用 --resume 断点续训）")


if __name__ == "__main__":
    main()
