from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4


def history_path() -> Path:
    root = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "BackgroundRemover"
    return root / "history.json"


class HistoryStore:
    def __init__(self, path: Path | None = None): self.path = path or history_path()
    def load(self) -> list[dict]:
        try: return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError): return []
    def add(self, entry: dict) -> None:
        rows = self.load(); rows.insert(0, {"timestamp": datetime.now().isoformat(timespec="seconds"), **entry})
        rows = rows[:100]; self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"); os.replace(tmp, self.path)
