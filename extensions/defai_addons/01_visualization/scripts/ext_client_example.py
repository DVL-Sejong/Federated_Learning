"""외부 연합 학습 환경 → 시각화 도구 REST 연동 예제.

2차년도 Flask 멀티 컴퓨팅 시뮬레이션처럼 별도 프로세스/컴퓨터에서 도는 연합 학습 환경이
라운드 기록을 대시보드로 전송하는 방법을 보여준다. 여기서는 내장 시뮬레이터를 '외부 환경'으로 삼아
매 라운드 기록을 HTTP POST 로 보낸다.

  1) python scripts/run_monitor.py --port 5000
  2) python scripts/ext_client_example.py --url http://127.0.0.1:5000
"""
import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "shared")))

from defai_addons.fl.recorder import RunRecorder, _jsonable
from defai_addons.monitor.simulator import HierarchicalFLSimulator


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(_jsonable(body)).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))


class RestRecorder(RunRecorder):
    """기록을 로컬에 쌓는 동시에 대시보드 서버로 전송."""

    def __init__(self, base, config):
        super().__init__(config, name="external")
        self.base, self.remote_id, self._sent_logs = base, None, 0

    def set_topology(self, topology, clients_static):
        super().set_topology(topology, clients_static)
        self.remote_id = post(self.base + "/api/ext/runs", {"name": "external-" + self.config.get("algorithm", ""),
                                                            "config": self.config, "topology": topology,
                                                            "clients": clients_static})["id"]
        # 2차년도 엔드포인트: 클라이언트 데이터 비율·엔트로피 계산
        for c in clients_static[:3]:
            r = post(self.base + "/api/ext/client_stats", {"client_id": c["id"], "n": c["n"], "class_counts": c["class_counts"]})
            print("client_stats", r["client_id"], "ratio=%.3f entropy=%.3f" % (r["ratio"], r["entropy"]))

    def add_round(self, rec):
        super().add_round(rec)
        body = dict(rec)
        body["logs"] = self.logs[self._sent_logs:]
        body["sketch"] = self.sketch
        self._sent_logs = len(self.logs)
        post(self.base + "/api/ext/runs/%s/rounds" % self.remote_id, body)

    def set_status(self, status, error=None, summary=None):
        super().set_status(status, error, summary)
        if status == "finished" and self.remote_id:
            post(self.base + "/api/ext/runs/%s/finish" % self.remote_id, {"summary": summary or {}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5000")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--algorithm", default="fedprox")
    a = ap.parse_args()
    cfg = {"rounds": a.rounds, "algorithm": a.algorithm, "n_malicious": 1, "sleep": 0.3}
    rec = RestRecorder(a.url.rstrip("/"), cfg)
    sim = HierarchicalFLSimulator(cfg, rec)
    sim.run()
    print("완료: 대시보드에서 실행 id %s 를 선택하세요" % rec.remote_id)


if __name__ == "__main__":
    main()
