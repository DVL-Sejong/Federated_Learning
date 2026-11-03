# 1-1 계층형 연합학습 시뮬레이터

보고서 위치: 3차년도 보고서 가-(1) · 패키지: `hierarchical_fl` · 소스: [`src/hierarchical_fl`](src/hierarchical_fl) · 테스트: [`tests/modules/test_hierarchical.py`](../../tests/modules/test_hierarchical.py)

클라이언트–엣지 서버–중앙 서버 3계층 연합학습을 한 대의 컴퓨터 또는 여러 프로세스/컴퓨터에서 실행하는 시뮬레이터입니다.
엣지 서버는 Flask REST 서버, 학습 메시지는 Flower `Message`(protobuf 직렬화)이며, 각 계층의 집계는 Flower `Strategy`(내장 또는 KIST `defai.strategies`)가 수행합니다.

## 연차별 개발 항목 통합

| 연차 | 개발 항목 | 이 모듈에서의 처리 |
|---|---|---|
| 1차 | 연합 학습 시험용 파이썬 환경 기반 단일 컴퓨터 시뮬레이션 환경 설계 (멀티스레딩·멀티프로세싱 병렬화, asyncio 비동기 업로드) | **유지·통합**: `transport=local`(한 프로세스, Flask test client로 실제 REST 경로 실행), `client-workers`(엣지 내부 클라이언트 학습 스레드 병렬화), `--spawn-edges`(엣지별 프로세스 병렬화), 엣지 호출은 `asyncio.gather` 비동기 통신 |
| 2차 | Flask 기반 멀티 컴퓨팅 시뮬레이션 환경 (Flower 라운드 오케스트레이션, RESTful API, SQLite DB 래퍼, 데이터 비율·정보 엔트로피 산출, 모니터링 대시보드) | **유지·통합**: REST API 엣지 서버, SQLite 레지스트리(`hfl.sqlite`), `/clients/stats` 엔드포인트의 데이터 비율 R=n_i/N·엔트로피 H=−Σp log₂p, 대시보드 R/H 히트맵·정확도/손실 추적. 엣지 1개·라운드 1회로 설정하면 단일 서버 구조 |
| 3차 | Flask 기반 계층적 연합학습 시뮬레이션 (엣지 서버 구성, 서버–엣지–클라이언트 REST 통신, 게임 이론 매칭, 계층 트리·우선순위·통신 상태·매칭 행렬 시각화, FedAvg/FedProx/q-FedAvg 비교) | **최신 구현**: 아래 구조 전체 |

## 구조

```text
HierarchicalServer (server.py)
 ├─ Flower Strategy.start(grid=EdgeGrid)              전역 집계 (fedavg | fedprox | qfedavg | fedmedian | krum ... 또는 KIST 전략)
 ├─ RoundHook.configure_train → prepare_round()        참여 상태 → 엣지 헬스체크(503) → 기대 시간·마감 → 게임 이론 매칭
 └─ EdgeGrid (edge_grid.py, flwr.serverapp.Grid)       asyncio로 엣지에 Flower Message 동시 전송
       │ POST /api/v1/sessions/<sid>/messages  (application/x-flower-messages, protobuf)
       ▼
EdgeSession (edge_server.py, Flask)                    edge-rounds회 엣지 집계 (HierFAVG)
 ├─ Flower Strategy (엣지 계층)                         configure_train / aggregate_train
 └─ InProcessGrid → ClientApp 핸들러                    내장 ClientApp 또는 KIST defai.client_app.train
```

* **게임 이론 매칭** (`matching.py`): 클라이언트 효용 `u_ij = E(R_ij) − cd_j`, `E(R_ij) = 보상_i · q̂_j · P(T_ij ≤ D)`, `cd_j` = 데이터 비용 + 통신 비용(라운드 중앙값 대비).
  엣지의 품질 추정량은 지수 평활(EMA, 망각 효과) `q̂_j ← β q̂_j + (1−β) q_j`, `q_j = (1 + cos(업데이트_j, 엣지 중앙값 업데이트))/2`.
  엣지 우선순위 = `q̂_j · P(T_ij ≤ D)`, 클라이언트 제안형 지연 수락(Gale–Shapley, 엣지 용량)으로 안정 매칭을 구하고 blocking pair 수를 기록합니다. 비교용 `static`(최근접)·`random` 매칭 제공.
* **네트워크 모델** (`network.py`): 단위 정사각형 위 위치, 지연 = 15 ms + 180 ms × 거리, 경로 대역폭 감쇠, 단말 속도 계수(log-normal). 학습은 실제로 수행하고 시간은 가상 시간(ms)으로 기록합니다. 마감 정책 `none | fixed | adaptive`(기대 시간 분위수 × 여유).
* **예외 처리**: 엣지 헬스체크 실패(HTTP 503) 시 매칭에서 제외되고 클라이언트가 다른 엣지에 재배정, 응답 지연(타임아웃)·클라이언트 중도 이탈은 Flower 오류 응답으로 처리.
* **SQLite 레지스트리** (`registry.py`): runs, clients(데이터 비율·엔트로피·역할), edges, rounds, assignments(효용·우선순위·q̂·상태), edge_rounds(통신량·시간), events(보안·장애 로그).
* **대시보드** (`dashboard.py`): 실험 목록·실행, 계층 트리, 매칭 행렬, 엣지 우선순위 rank, 엣지 통신 상태, Accuracy/Loss, 라운드별 통신량·가상 시간, R/H/레이블 분포 히트맵, 클라이언트 학습 추이와 이상치(training trend / outlier detection), 참여·예외 처리 표와 이벤트 로그.

## 실행

```bash
python -m hierarchical_fl run --set num-clients=12 --set num-edges=3 --set global-rounds=5
python -m hierarchical_fl run --spawn-edges --set edge-strategy=fedprox
python -m hierarchical_fl run --set backend=kist --set backend-config.dataset=synthetic
python -m hierarchical_fl run --set matching.mode=static --set deadline.policy=adaptive --set backend-config.partitioner=dirichlet
python -m hierarchical_fl dashboard --results results
```

여러 컴퓨터에서 실행할 때는 엣지마다 `python -m hierarchical_fl edge --edge-id 0 --host 0.0.0.0 --port 6101`을 띄우고
전역 서버에서 `python -m hierarchical_fl run --edges http://host1:6101,http://host2:6101,http://host3:6101`로 연결합니다.

주요 설정(`config.py`): `backend`, `backend-config`(dataset, partitioner, dirichlet-alpha, local-epochs, learning-rate ...), `num-clients`, `num-edges`, `edge-capacity`, `global-rounds`, `edge-rounds`, `edge-strategy`, `global-strategy`, `matching.*`, `deadline.*`, `transport`(local|http), `client-workers`, `plugins`.

## REST API (엣지 서버)

| 메서드 | 경로 | 내용 |
|---|---|---|
| GET | `/api/v1/health` | 상태, 세션, 요청 수, 송수신 바이트, CPU·메모리 |
| POST | `/api/v1/sessions` | 실험 세션 생성 (백엔드, 클라이언트 수, 엣지 전략, 플러그인) |
| POST | `/api/v1/sessions/<sid>/messages` | Flower Message(TRAIN/EVALUATE) 처리 후 응답 Message 반환 |
| GET | `/api/v1/sessions/<sid>/health?round=r` | 장애 주입 시 503 |
| POST | `/api/v1/sessions/<sid>/fault` | 장애 주입 (down/hang 라운드) |
| GET | `/api/v1/sessions/<sid>/clients/stats` | 클라이언트 데이터 수, 레이블 분포, 데이터 비율, 정보 엔트로피 |
| GET | `/api/v1/sessions/<sid>/history` | 엣지 라운드 보고 |
| DELETE | `/api/v1/sessions/<sid>` | 세션 삭제 |

## KIST 시스템과 연결

`backend=kist`로 실행하면 각 엣지 서버가 KIST의 `defai.client_app.train/evaluate`로 클라이언트를 학습시키고, 엣지·전역 집계에 `defai.strategies.build_strategy`가 만든 KIST 전략을 씁니다. 모델·데이터 분할·run config 검증도 KIST 코드를 그대로 사용합니다.

```bash
python -m hierarchical_fl run --set backend=kist --set edge-strategy=fedprox --set backend-config.dataset=synthetic
```

## 출력

`<output-dir>/<run-id>/`에 `hfl.sqlite`, `summary.json`(라운드별 정확도·손실·가상 시간·통신량·매칭 정보·플러그인 요약), `network.json`, (`--spawn-edges` 시) `edge_logs/`가 생성됩니다.

## 검증

* `tests/modules/test_hierarchical.py`: 지연 수락 매칭의 안정성(blocking pair 0), 2계층 실행, SQLite 기록, 대시보드 API.
* in-process(Flask test client)와 HTTP 엣지 프로세스 모두에서 내장 백엔드·KIST 백엔드(KIST ClientApp + KIST FedProx 엣지 집계 + KIST FedAvg 전역 집계)를 실행해 확인했습니다.

## 측정 결과

측정 환경: `scripts/run_demo.py` 기본 설정(synthetic 데이터, 3라운드 내외의 짧은 실행, CPU i9-11900F). 짧은 실행이므로 절대값보다 같은 설정 안의 비교로 해석합니다.

| 실행 | 전송 방식 | 최종 정확도 | 최근 3R 평균 | 가상 시간 합계 | 전송량 |
|---|---|---|---|---|---|
| 클라이언트 9, 엣지 3, 3라운드, FedAvg | in-process Flask | 0.9065 | 0.778 | 123.1 s | 114.2 MB |
| 같은 설정 | 엣지 3개 별도 프로세스(HTTP) | 0.9050 | 0.776 | 123.1 s | 114.2 MB |

두 전송 방식의 라운드별 결과가 사실상 같아 REST 경로가 학습 결과를 바꾸지 않음을 확인했습니다(차이는 스레드 수에 따른 부동소수점 누적 순서). KIST 백엔드(엣지 FedProx·q-FedAvg, 전역 FedAvg)도 같은 경로로 실행됩니다.
