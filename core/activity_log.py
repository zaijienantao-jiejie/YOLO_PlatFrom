# -*- coding: utf-8 -*-
"""活动日志：用户操作审计（JSONL）"""
import json
import time
from pathlib import Path
from typing import List


class ActivityLogger:
    def __init__(self, log_dir: Path):
        self.file = Path(log_dir) / "activity.jsonl"
        self.file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, action: str, target: str = "", detail: str = "", actor: str = ""):
        entry = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "actor": actor or "system",
            "action": action,
            "target": target,
            "detail": detail,
        }
        try:
            with self.file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def read(self, limit: int = 500, reverse: bool = True) -> List[dict]:
        if not self.file.exists():
            return []
        lines = self.file.read_text(encoding="utf-8").splitlines()
        if reverse:
            lines = lines[-limit:][::-1]
        else:
            lines = lines[-limit:]
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
