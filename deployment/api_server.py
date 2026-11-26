# -*- coding: utf-8 -*-
"""FastAPI 推理服务：/predict /predict/batch /predict/stream /health /info /metrics"""
import time
from typing import List

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from core import logger
from core.dependencies import resolve_device


def _detect_payload(res, task: str, inference_time: float):
    boxes = res.boxes
    out = []
    if boxes is None:
        return out
    for cls, xyxy, xywhn, conf in zip(boxes.cls.tolist(), boxes.xyxy.tolist(), boxes.xywhn.tolist(), boxes.conf.tolist()):
        item = {
            "class": res.names[int(cls)],
            "class_id": int(cls),
            "confidence": round(float(conf), 4),
            "bbox": [round(float(v), 1) for v in xyxy],
            "bbox_normalized": [round(float(v), 4) for v in xywhn],
        }
        if task == "segment" and res.masks is not None:
            item["polygon"] = [[round(float(x), 4), round(float(y), 4)] for x, y in res.masks.xyn[0]] if len(res.masks.xyn) else []
        if task == "pose" and res.keypoints is not None:
            item["keypoints"] = [{"x": round(float(k[0]), 4), "y": round(float(k[1]), 4),
                                  "confidence": round(float(k[2]), 4), "visible": float(k[2]) >= 0.5}
                                 for k in res.keypoints.xyn[0]]
        if task == "obb" and res.obb is not None:
            import math
            cx, cy, w, h, rad = res.obb.xywhr[0].tolist()
            item["angle"] = round(math.degrees(rad) % 180, 2)
        out.append(item)
    return out


def create_app(model_file: str, task: str = "detect", conf: float = 0.25, iou: float = 0.45,
               device: str = "auto") -> FastAPI:
    from ultralytics import YOLO
    model = YOLO(model_file)
    device = resolve_device(device)
    app = FastAPI(title="YOLO Platform API", version="2.0.0",
                  description=f"YOLO Platform 推理服务（{task} 任务，模型: {model_file}）")
    stats = {"requests": 0, "total_ms": 0.0, "started": time.time()}

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_file, "task": task, "device": device,
                "uptime_seconds": round(time.time() - stats["started"], 1)}

    @app.get("/info")
    def info():
        return {"model": model_file, "task": task, "classes": model.names,
                "imgsz": getattr(model.model, "args", None) and model.model.args.get("imgsz") or 640}

    @app.get("/metrics")
    def metrics():
        avg = round(stats["total_ms"] / stats["requests"], 2) if stats["requests"] else 0.0
        return {"requests": stats["requests"], "avg_inference_ms": avg,
                "uptime_seconds": round(time.time() - stats["started"], 1)}

    def _predict_bytes(data: bytes):
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(data)).convert("RGB")
        t0 = time.time()
        results = model.predict(img, conf=conf, iou=iou, device=device, verbose=False)
        dt = time.time() - t0
        stats["requests"] += 1
        stats["total_ms"] += dt * 1000
        r = results[0]
        if task == "classify" and r.probs is not None:
            top5_idx = r.probs.top5.tolist()
            predictions = [{"class": r.names[int(i)], "class_id": int(i),
                            "confidence": round(float(r.probs.top5conf.tolist()[k]), 4)}
                           for k, i in enumerate(top5_idx)]
        else:
            predictions = _detect_payload(r, task, dt)
        return {
            "success": True,
            "data": {
                "predictions": predictions,
                "inference_time": round(dt, 4),
                "model": model_file,
            },
        }

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        try:
            data = await file.read()
            return _predict_bytes(data)
        except Exception as e:
            logger.get_logger().error(f"/predict 错误: {e}")
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    @app.post("/predict/batch")
    async def predict_batch(files: List[UploadFile] = File(...)):
        try:
            results = []
            for f in files:
                data = await f.read()
                results.append({"file": f.filename, **_predict_bytes(data)})
            return {"success": True, "data": results}
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    @app.post("/predict/stream")
    async def predict_stream(file: UploadFile = File(...), frame_stride: int = 1):
        """上传视频文件，返回每帧（抽样）检测结果"""
        import cv2
        import tempfile
        from pathlib import Path
        try:
            data = await file.read()
            suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            cap = cv2.VideoCapture(tmp_path)
            frames_out = []
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % max(frame_stride, 1) == 0:
                    t0 = time.time()
                    results = model.predict(frame, conf=conf, iou=iou, device=device, verbose=False)
                    dt = time.time() - t0
                    stats["requests"] += 1
                    stats["total_ms"] += dt * 1000
                    frames_out.append({"frame_index": idx,
                                       "predictions": _detect_payload(results[0], task, dt),
                                       "inference_time": round(dt, 4)})
                idx += 1
            cap.release()
            return {"success": True, "data": {"total_frames": idx, "frames": frames_out}}
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    return app
