"""연합 학습 환경 평가 시각화 도구 - Flask 웹 대시보드 서버.

실행:  python -m defai_addons.monitor.server --port 5000   (또는 scripts/run_monitor.py)

REST API
  GET  /api/defaults                       시뮬레이터/비교 기본 설정
  GET  /api/runs                           실행 목록
  POST /api/runs                           시뮬레이션 실행 {config, name?}
  POST /api/runs/compare                   알고리즘 비교 실행 {config, algorithms:[...], seeds: n}
  GET  /api/runs/<id>?since=k              실행 상태 + 시각화 뷰
  POST /api/runs/<id>/stop                 중지
  POST /api/runs/<id>/save                 runs/<id>.json 저장
  GET  /api/groups/<gid>                   알고리즘 비교 뷰
  POST /api/comparisons                    지역 분산 vs 연합 비교 실행
  GET  /api/comparisons[/<id>]             비교 진행 상황/보고서 (실시간)

외부 연합 학습 환경 연동 (2차년도 Flask 멀티 컴퓨팅 시뮬레이션의 RESTful 통신 방식)
  POST /api/ext/runs                       {config, topology, clients} → {id}
  POST /api/ext/runs/<id>/rounds           라운드 기록 1건 (simulator 와 같은 스키마)
  POST /api/ext/runs/<id>/finish
  POST /api/ext/client_stats               {client_id, n, class_counts} → 데이터 비율·정보 엔트로피
"""
import argparse
import glob
import os
import threading
import uuid
from typing import Dict, List, Optional

from flask import Flask, abort, jsonify, render_template, request

from ..common.metrics import data_ratio, entropy
from ..compare.runner import DEFAULTS as CMP_DEFAULTS
from ..compare.runner import ENV_LABELS, ComparisonJob
from ..fl.algorithms import ALGORITHMS
from ..fl.recorder import RunRecorder, _jsonable
from ..monitor.simulator import DEFAULTS as SIM_DEFAULTS
from ..monitor.simulator import HierarchicalFLSimulator
from .analytics import compare_runs, compute_views


class RunManager:
    def __init__(self, store_dir: Optional[str] = None, max_parallel: int = 3):
        self.runs: Dict[str, RunRecorder] = {}
        self.sims: Dict[str, HierarchicalFLSimulator] = {}
        self.jobs: Dict[str, ComparisonJob] = {}
        self.client_stats: Dict[str, Dict] = {}
        self.store_dir = store_dir
        self._views_cache: Dict[str, tuple] = {}
        self._sem = threading.Semaphore(max_parallel)
        self._lock = threading.Lock()
        if store_dir:
            os.makedirs(store_dir, exist_ok=True)
            for p in sorted(glob.glob(os.path.join(store_dir, "*.json"))):
                try:
                    r = RunRecorder.load(p)
                    self.runs[r.id] = r
                except Exception:
                    pass

    # ---- simulation runs ------------------------------------------------
    def start(self, config: Dict, name: Optional[str] = None, group: Optional[str] = None) -> RunRecorder:
        rec = RunRecorder(config, name=name or config.get("algorithm", "run"), group=group)
        sim = HierarchicalFLSimulator(config, rec)
        rec.config = _jsonable(sim.cfg)
        with self._lock:
            self.runs[rec.id] = rec
            self.sims[rec.id] = sim

        def work():
            with self._sem:
                try:
                    sim.run()
                except Exception:
                    pass
                if self.store_dir:
                    rec.save(os.path.join(self.store_dir, rec.id + ".json"))
        threading.Thread(target=work, daemon=True).start()
        return rec

    def views(self, rid: str) -> Dict:
        rec = self.runs[rid]
        cached = self._views_cache.get(rid)
        if cached and cached[0] == rec.version:
            return cached[1]
        v = _jsonable(compute_views(rec.snapshot(0)))
        self._views_cache[rid] = (rec.version, v)
        return v

    def group(self, gid: str) -> List[RunRecorder]:
        return [r for r in self.runs.values() if r.group == gid]

    # ---- local-distributed vs federated ---------------------------------
    def start_comparison(self, config: Dict) -> ComparisonJob:
        job = ComparisonJob(config)
        self.jobs[job.id] = job

        def work():
            with self._sem:
                try:
                    job.run()
                except Exception:
                    pass
        threading.Thread(target=work, daemon=True).start()
        return job


def create_app(store_dir: Optional[str] = None) -> Flask:
    here = os.path.dirname(os.path.abspath(__file__))
    app = Flask(__name__, template_folder=os.path.join(here, "templates"),
                static_folder=os.path.join(here, "static"))
    app.config["JSON_AS_ASCII"] = False
    app.config["TEMPLATES_AUTO_RELOAD"] = True  # 대시보드 템플릿 수정 시 서버 재시작 없이 반영
    mgr = RunManager(store_dir)
    app.extensions["defai_manager"] = mgr

    def get_run(rid: str) -> RunRecorder:
        if rid not in mgr.runs:
            abort(404)
        return mgr.runs[rid]

    @app.route("/")
    def index():
        return render_template("dashboard.html")

    @app.route("/api/defaults")
    def defaults():
        return jsonify({"simulation": _jsonable(SIM_DEFAULTS), "algorithms": list(ALGORITHMS),
                        "comparison": _jsonable(CMP_DEFAULTS), "env_labels": ENV_LABELS})

    @app.route("/api/runs", methods=["GET"])
    def list_runs():
        return jsonify(sorted([r.meta() for r in mgr.runs.values()], key=lambda m: -m["created"]))

    @app.route("/api/runs", methods=["POST"])
    def start_run():
        body = request.get_json(force=True) or {}
        rec = mgr.start(body.get("config", {}), name=body.get("name"))
        return jsonify(rec.meta())

    @app.route("/api/runs/compare", methods=["POST"])
    def start_algorithm_comparison():
        body = request.get_json(force=True) or {}
        gid = "g" + uuid.uuid4().hex[:6]
        metas = []
        base = dict(body.get("config", {}))
        seed0 = int(base.get("seed") or 0)
        # 같은 seed 끼리 알고리즘을 짝지어 비교할 수 있도록 seed 를 바깥 반복으로 둔다
        for k in range(max(1, int(body.get("seeds", 1)))):
            for alg in body.get("algorithms", list(ALGORITHMS)):
                cfg = dict(base, algorithm=alg, seed=seed0 + k)
                metas.append(mgr.start(cfg, name="%s·s%d" % (alg, seed0 + k), group=gid).meta())
        return jsonify({"group": gid, "runs": metas})

    @app.route("/api/runs/<rid>")
    def run_state(rid):
        rec = get_run(rid)
        since = int(request.args.get("since", 0))
        snap = rec.snapshot(since)
        snap["meta"] = rec.meta()
        if request.args.get("views", "1") == "1":
            snap["views"] = mgr.views(rid)
        return jsonify(snap)

    @app.route("/api/runs/<rid>/stop", methods=["POST"])
    def stop_run(rid):
        get_run(rid)
        if rid in mgr.sims:
            mgr.sims[rid].stop()
        return jsonify({"ok": True})

    @app.route("/api/runs/<rid>/save", methods=["POST"])
    def save_run(rid):
        rec = get_run(rid)
        d = mgr.store_dir or os.path.join(os.getcwd(), "runs")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, rid + ".json")
        rec.save(path)
        return jsonify({"path": path})

    @app.route("/api/groups/<gid>")
    def group_view(gid):
        runs = mgr.group(gid)
        if not runs:
            abort(404)
        return jsonify(_jsonable(compare_runs([r.full() for r in runs])))

    # ---- comparison jobs -------------------------------------------------
    @app.route("/api/comparisons", methods=["GET"])
    def list_comparisons():
        return jsonify([{"id": j.id, "status": j.status, "created": j.created} for j in
                        sorted(mgr.jobs.values(), key=lambda j: -j.created)])

    @app.route("/api/comparisons", methods=["POST"])
    def start_comparison():
        body = request.get_json(force=True) or {}
        job = mgr.start_comparison(body.get("config", {}))
        return jsonify({"id": job.id})

    @app.route("/api/comparisons/<jid>")
    def comparison_state(jid):
        if jid not in mgr.jobs:
            abort(404)
        return jsonify(mgr.jobs[jid].state())

    # ---- external ingestion ---------------------------------------------
    @app.route("/api/ext/runs", methods=["POST"])
    def ext_create():
        body = request.get_json(force=True) or {}
        rec = RunRecorder(body.get("config", {}), name=body.get("name", "external"), group=body.get("group"))
        rec.set_topology(body.get("topology", {}), body.get("clients", []))
        rec.set_status("running")
        mgr.runs[rec.id] = rec
        return jsonify({"id": rec.id})

    @app.route("/api/ext/runs/<rid>/rounds", methods=["POST"])
    def ext_round(rid):
        rec = get_run(rid)
        body = request.get_json(force=True) or {}
        for l in body.pop("logs", []):
            rec.log(l.get("round", body.get("round", 0)), l.get("level", "info"), l.get("event", "log"),
                    client=l.get("client"))
        if "sketch" in body:
            rec.set_sketch({int(k): v for k, v in body.pop("sketch").items()})
        rec.add_round(body)
        return jsonify({"ok": True, "n_rounds": len(rec.rounds)})

    @app.route("/api/ext/runs/<rid>/finish", methods=["POST"])
    def ext_finish(rid):
        rec = get_run(rid)
        rec.set_status("finished", summary=(request.get_json(silent=True) or {}).get("summary"))
        return jsonify({"ok": True})

    @app.route("/api/ext/client_stats", methods=["POST"])
    def ext_client_stats():
        """2차년도 '모델 데이터 처리를 위한 API 엔드포인트': 데이터 비율 p_i 와 엔트로피 H_i 계산."""
        body = request.get_json(force=True) or {}
        cid = str(body["client_id"])
        counts = body.get("class_counts") or []
        mgr.client_stats[cid] = {"n": int(body.get("n", sum(counts))), "class_counts": counts}
        ids = sorted(mgr.client_stats)
        ratio = data_ratio([mgr.client_stats[i]["n"] for i in ids])
        return jsonify({"client_id": cid, "ratio": float(ratio[ids.index(cid)]),
                        "entropy": entropy(counts) if counts else 0.0,
                        "all": {i: {"ratio": float(r), "entropy": entropy(mgr.client_stats[i]["class_counts"])
                                    if mgr.client_stats[i]["class_counts"] else 0.0}
                                for i, r in zip(ids, ratio)}})

    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description="deFAI 연합 학습 환경 평가 시각화 도구")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--store", default="runs", help="실행 기록 저장 폴더 (시작 시 자동 로드)")
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    app = create_app(a.store)
    print("deFAI monitor → http://%s:%d" % (a.host, a.port))
    app.run(host=a.host, port=a.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
