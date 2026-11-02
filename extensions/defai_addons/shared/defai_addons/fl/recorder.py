"""실행 기록기: 시뮬레이터가 라운드마다 남기는 기록을 스레드 안전하게 보관하고
대시보드가 증분(since=k) 폴링으로 가져갈 수 있게 한다.

외부 연합 학습 환경(예: 2차년도 Flask 멀티 컴퓨팅 시뮬레이션)도 REST 로 같은 형식의
라운드 기록을 보내면 동일한 시각화를 사용할 수 있다 (monitor/server.py 참조).
"""
import json
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (f != f or f in (float("inf"), float("-inf"))) else round(f, 6)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


class RunRecorder:
    def __init__(self, config: Dict[str, Any], name: Optional[str] = None, group: Optional[str] = None,
                 run_id: Optional[str] = None):
        self.id = run_id or uuid.uuid4().hex[:8]
        self.name = name or config.get("algorithm", "run")
        self.group = group
        self.config = _jsonable(config)
        self.created = time.time()
        self.status = "created"
        self.error: Optional[str] = None
        self.topology: Dict[str, Any] = {}
        self.clients_static: List[Dict[str, Any]] = []
        self.rounds: List[Dict[str, Any]] = []
        self.logs: List[Dict[str, Any]] = []
        self.sketch: Dict[int, List[float]] = {}  # 클라이언트 누적 업데이트 방향의 저차원 스케치
        self.summary: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self.version = 0

    # ---- writers -------------------------------------------------------
    def set_topology(self, topology: Dict[str, Any], clients_static: List[Dict[str, Any]]):
        with self._lock:
            self.topology = _jsonable(topology)
            self.clients_static = _jsonable(clients_static)
            self.version += 1

    def add_round(self, rec: Dict[str, Any]):
        with self._lock:
            self.rounds.append(_jsonable(rec))
            self.version += 1

    def log(self, rnd: int, level: str, event: str, client: Optional[int] = None, **kw):
        with self._lock:
            self.logs.append(_jsonable({"round": rnd, "t": time.time(), "level": level, "event": event,
                                        "client": client, **kw}))

    def set_sketch(self, sketch: Dict[int, np.ndarray]):
        with self._lock:
            self.sketch = {int(k): _jsonable(v) for k, v in sketch.items()}

    def set_status(self, status: str, error: Optional[str] = None, summary: Optional[Dict] = None):
        with self._lock:
            self.status = status
            self.error = error
            if summary is not None:
                self.summary = _jsonable(summary)
            self.version += 1

    # ---- readers -------------------------------------------------------
    def meta(self) -> Dict[str, Any]:
        with self._lock:
            return {"id": self.id, "name": self.name, "group": self.group, "status": self.status,
                    "error": self.error, "created": self.created, "n_rounds": len(self.rounds),
                    "total_rounds": self.config.get("rounds"), "algorithm": self.config.get("algorithm"),
                    "summary": self.summary}

    def snapshot(self, since: int = 0) -> Dict[str, Any]:
        with self._lock:
            return {"meta": None, "config": self.config, "topology": self.topology,
                    "clients": self.clients_static, "rounds": self.rounds[since:],
                    "logs": [l for l in self.logs if l["round"] >= since], "sketch": self.sketch,
                    "since": since, "n_rounds": len(self.rounds), "status": self.status,
                    "error": self.error, "summary": self.summary}

    def full(self) -> Dict[str, Any]:
        s = self.snapshot(0)
        s["meta"] = self.meta()
        return s

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.full(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "RunRecorder":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        m = d.get("meta") or {}
        r = cls(d.get("config", {}), name=m.get("name"), group=m.get("group"), run_id=m.get("id"))
        r.topology, r.clients_static = d.get("topology", {}), d.get("clients", [])
        r.rounds, r.logs = d.get("rounds", []), d.get("logs", [])
        r.sketch = {int(k): v for k, v in d.get("sketch", {}).items()}
        r.status, r.summary = d.get("status", "finished"), d.get("summary", {})
        return r
