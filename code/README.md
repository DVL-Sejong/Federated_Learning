# 세종대학교 deFAI 공개 모듈 (1~3차년도 통합본)

세종대학교가 1~3차년도에 개발한 연합학습 시뮬레이션·보안·경량화 기술을 **기능별 모듈 7개**로 통합한 코드입니다.
차년도별 복사본을 두지 않고, 후속 차년도에서 발전·대체된 기능은 최신 구현만 남겼으며 대체되지 않은 기존 기능은 그대로 유지했습니다.
모든 모듈은 **Flower 1.42 Message API**(`ServerApp`/`ClientApp`/`Strategy`/`Grid`)와 **Flask**로 구현했고,
KIST 연합학습 시스템([rixile9999/deFAI](https://github.com/rixile9999/deFAI), 연동 기준 커밋 `b21d628`)의
ClientApp·전략·모델·데이터에 그대로 붙여서 실행할 수 있습니다.

| 폴더 | 모듈 | 통합한 개발 항목 |
|---|---|---|
| [`hierarchical_fl`](hierarchical_fl/README.md) | 계층형 연합학습 시뮬레이터 | (1차) 파이썬 단일 컴퓨터 시뮬레이션 → (2차) Flask 멀티 컴퓨팅 시뮬레이션 → (3차) Flask 계층적 연합학습 시뮬레이션 |
| [`security_sim`](security_sim/README.md) | 보안 공격·방어 시뮬레이션 | (1차) 보안 시험 환경·네트워크 비용 산출 → (2차) 다계층 보안 시험 → (3차) 보안 시뮬레이션 고도화 |
| [`scenario_test`](scenario_test/README.md) | 연합학습 시나리오 테스트 | (3차) 다양한 연합학습 시나리오 테스트 |
| [`graph_pruning`](graph_pruning/README.md) | 연산 그래프 기반 모델 가지치기 | (1차) 소규모화 설계 → (2차) 고도화된 소규모화 → (3차) 연산 그래프 기반 가지치기 |
| [`layer_kd`](layer_kd/README.md) | 중간층 매칭 기반 지식 증류 | (1차) 지식 증류 설계 → (2차) 중간 레이어 매칭 → (3차) 고도화된 중간층 매칭 |
| [`data_imbalance`](data_imbalance/README.md) | 데이터 불균형 해결 | (1차) 데이터 불균형 해결 기술 설계 (조합 샘플링·동적 가중치) |
| [`network_cost`](network_cost/README.md) | 다계층 네트워크 비용 최적화 | (2차) 다계층 네트워크 비용 최적화 시뮬레이션 |
| `fl_core` | 공통 기반 | Flower Grid/전략 래퍼, 플러그인 인터페이스, 내장/KIST 백엔드, Flower 메시지 직렬화, KIST용 ServerApp/ClientApp |

## 폴더 구조

```text
code/
├── README.md, pyproject.toml, requirements.txt, run_demo.py
├── fl_core/            공통: backends.py(builtin|kist), grid.py, plugins.py, strategies.py, wire.py, kist_apps.py, task.py
├── hierarchical_fl/    Flask 엣지 서버, 전역 서버, 게임 이론 매칭, SQLite 레지스트리, 대시보드
├── security_sim/       공격 10종, 방어 9종, 모델 역추적 공격 API 실험
├── scenario_test/      S1~S5 예외 시나리오, 적응력 지수, 클라이언트 기여도
├── graph_pruning/      torch.fx 연산 그래프, 경로 중요도, 동적 임계값, BN-L1 희소성, 연합 가지치기
├── layer_kd/           로짓·앙상블·가중 특징맵·어텐션·GAN 중간층 매칭 증류, 연합 증류
├── data_imbalance/     SMOTE·Tomek·ENN 조합 샘플링, 데이터량·손실 기반 동적 가중치
├── network_cost/       엣지 컴퓨팅(업링크 압축), LRU 캐시, DAG 비용 최적 경로
└── tests/              pytest 24개 (KIST 연동 테스트 포함)
```

## 설치

검증 환경: Windows 11, Python 3.12.7, CPU 전용(i9-11900F), torch 2.9.1+cpu, flwr 1.42.0, Flask 3.1.3.

```bash
cd code
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e .
```

KIST 시스템과 연동할 때는 같은 환경에 KIST `defai`를 함께 설치합니다. KIST는 `torch<2.7`을 요구하지만 위 환경(torch 2.9.1)에서 KIST 코드 경로를 모두 실행해 확인했으므로 의존성 재설치 없이 `--no-deps`로 설치합니다.

```bash
git clone https://github.com/rixile9999/deFAI ../../deFAI
.venv/Scripts/python -m pip install -e ../../deFAI --no-deps
.venv/Scripts/python -m pip install "flwr-datasets[vision]==0.6.1" pillow
```

`pip install -e .` 대신 `PYTHONPATH=code`로 실행해도 됩니다. 데이터는 기본적으로 오프라인 `synthetic`(MNIST 형식 10클래스)을 쓰며 `--set dataset=mnist`(또는 `fashion-mnist`, `cifar10`)를 지정하면 torchvision으로 `code/data/`에 내려받습니다(`FL_DATA_DIR`로 변경).

## 바로 검토하기

```bash
.venv/Scripts/python -m pytest -q
.venv/Scripts/python run_demo.py
.venv/Scripts/python -m hierarchical_fl dashboard --results results/demo
```

* `pytest`: 7개 모듈과 KIST 연동(ClientApp·전략·ServerApp 래핑)을 모두 실행하는 24개 테스트 (약 1.5분). KIST `defai`가 없으면 연동 테스트 3개만 건너뜁니다.
* `run_demo.py`: 모듈별 CLI 11단계를 차례로 실행하고 결과(JSON/CSV/PNG/SQLite)를 `results/demo/`에 저장합니다. `--backend kist`로 KIST 백엔드, `--full`로 더 긴 설정, `--only 1,5`로 일부 단계만 실행합니다.
* 대시보드: `http://127.0.0.1:5000` 에서 계층 트리, 매칭 행렬, 우선순위, 엣지 통신 상태, 정확도/손실, 데이터 비율·엔트로피, 보안 이벤트를 확인하고 새 실험을 실행할 수 있습니다.

## KIST 시스템 연동

세 가지 방식으로 붙일 수 있으며, 어느 경우에도 KIST 저장소 파일은 수정하지 않습니다.

**1. 백엔드 교체**: 모든 실행기에서 `backend=kist`를 지정하면 KIST의 `defai.client_app.train/evaluate`(ClientApp 핸들러), `defai.strategies.build_strategy`(15개 전략), `defai.task.build_model`·데이터 분할·run config 검증을 그대로 사용합니다.

```bash
python -m hierarchical_fl run --set backend=kist --set edge-strategy=fedprox --set backend-config.dataset=synthetic
python -m security_sim run --backend kist --attack sign_flip --defense stat,hmac
python -m graph_pruning federated --backend kist
python run_demo.py --backend kist
```

**2. Python API (플러그인)**: 각 모듈은 `fl_core.plugins.Plugin`을 구현하며, Flower `Strategy` 래퍼와 ClientApp 미들웨어로 KIST 구성요소를 감쌉니다.

```python
from fl_core.backends import get_backend
from fl_core.plugins import run_flat
from security_sim import SecurityPlugin
from data_imbalance.plugin import ImbalancePlugin
from network_cost.plugin import NetworkCostPlugin

backend = get_backend("kist", {"dataset": "mnist-mirror", "partitioner": "dirichlet", "dirichlet-alpha": 0.3})
out = run_flat(backend, [SecurityPlugin({"attack": "insider_collusion", "defense": ["monitor", "hmac"]}),
                         ImbalancePlugin({"sampling": "none", "weighting": "size_loss"}),
                         NetworkCostPlugin({"optimizations": ["edge_compute", "cache", "dag"]})],
               num_clients=10, rounds=10, strategy="fedprox")
```

계층형 시뮬레이터에도 같은 플러그인을 넘깁니다: `HierarchicalServer({"backend": "kist", ...}, plugins=[...]).run()`.

**3. Flower 배포(`flwr run`)**: `fl_core.kist_apps`는 KIST의 ServerApp/ClientApp 동작(설정 검증, 전략 생성, 중앙 평가, `history.json` 기록)을 유지하면서 플러그인을 적용하는 `server_app`·`client_app`을 제공합니다. KIST `pyproject.toml`의 컴포넌트와 run config에 다음을 지정합니다.

```toml
[tool.flwr.app.components]
serverapp = "fl_core.kist_apps:server_app"
clientapp = "fl_core.kist_apps:client_app"

[tool.flwr.app.config]
sejong-plugins = '[{"target": "security_sim.plugin:SecurityPlugin", "config": {"attack": "none", "defense": ["hmac", "stat"]}}]'
```

환경변수 `SEJONG_PLUGINS`로도 지정할 수 있습니다. 이 앱의 `server_main`/`client_train`은 테스트(`tests/test_kist_integration.py`)에서 Flower Grid로 직접 실행해 검증했으며, Ray 기반 `flwr run` 시뮬레이션 엔진 자체는 이번 환경에서 실행하지 않았습니다.

## 플러그인 인터페이스

```python
class Plugin:
    def configure(self, cfg): ...                       계층형 실행 설정 수정 (예: 데이터 분할)
    def setup(self, owner): ...                         전역 서버 / 엣지 세션 / 단일 서버에서 초기화
    def wrap_strategy(self, strategy, tier, owner): ... tier = "flat" | "edge" | "global", Flower Strategy 래핑
    def client_middleware(self, owner): ...             ClientApp 핸들러 전후 처리 (서명, 공격, 압축 등)
    def train_hooks(self, owner): ...                   data_transform / loss_factory / report_fn
    def round_state(self, server, rnd, state): ...      참여·지연·엣지 장애 등 라운드 상태
    def edge_reply(self, session, rnd, content): ...    엣지 서버 응답 처리
    def transport(self, server, rnd, edge_id, reply):   엣지→전역 전송 구간
    def on_round_end(self, owner, rnd, row): ...        라운드 기록
```

플러그인은 `{"target": "module:Class", "config": {...}}` 형식으로 직렬화되어 Flask 엣지 프로세스와 Flower ClientApp에서도 같은 설정으로 생성됩니다.

## 결과와 한계

* 모든 학습·집계·평가는 실제로 수행됩니다. 통신 지연·대역폭·경로 비용은 네트워크 모델 위의 가상 시간과 비용이며, 바이트 수는 실제 Flower 메시지(protobuf) 크기로 측정합니다.
* 기본 `synthetic` 데이터는 내려받기 없이 동작 확인용이며, 성능 비교는 `mnist`/`fashion-mnist`/`cifar10`으로 수행하는 것을 권장합니다.
* 각 모듈 README에 이 컴퓨터에서 측정한 실행 결과와 재현 명령을 기록했습니다.
