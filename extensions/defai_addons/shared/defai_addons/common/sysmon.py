"""CPU/메모리 사용량 샘플러 (1차년도 '리소스 모니터링: CPU, 메모리 사용량 range line').

백그라운드 스레드에서 주기적으로 현재 프로세스의 CPU% 와 RSS 를 수집하고,
라운드 단위로 min / mean / max 를 요약해 range-line 시각화에 사용한다.
"""
import threading
import time
from typing import Dict, List

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


class ResourceSampler:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self._samples: List[Dict[str, float]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._proc = psutil.Process() if psutil else None

    def start(self) -> "ResourceSampler":
        if self._proc is None:
            return self
        self._proc.cpu_percent(None)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            try:
                s = {"t": time.time(), "cpu": self._proc.cpu_percent(None),
                     "mem": self._proc.memory_info().rss / 2 ** 20,
                     "sys_cpu": psutil.cpu_percent(None), "sys_mem": psutil.virtual_memory().percent}
                with self._lock:
                    self._samples.append(s)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def drain(self) -> Dict[str, float]:
        """마지막 drain 이후의 샘플을 요약하고 버퍼를 비운다."""
        with self._lock:
            s, self._samples = self._samples, []
        if not s:
            return {}
        out = {}
        for k in ("cpu", "mem", "sys_cpu", "sys_mem"):
            v = [x[k] for x in s]
            out[k + "_min"], out[k + "_mean"], out[k + "_max"] = min(v), sum(v) / len(v), max(v)
        out["n"] = len(s)
        return out
