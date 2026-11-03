# 세종대학교 deFAI 연합학습 연구 코드

IITP 「분산형 데이터 연합컴퓨팅 기술(deFAI) 개발 및 실증」 과제에서 세종대학교가 1~3차년도에 개발한 코드를 한곳에 모은 저장소입니다.
폴더는 3차년도 연차보고서의 세종대학교 연구 항목 순서를 따릅니다. 차년도별 복사본은 두지 않았습니다. 후속 차년도에서 발전한 기능은 최신 구현만 남겼고, 대체되지 않은 이전 기능은 그대로 유지했습니다.

모든 코드는 하나의 파이썬 환경에서 함께 설치되고, KIST 연합학습 시스템([rixile9999/deFAI](https://github.com/rixile9999/deFAI))과 같은 Flower 1.42 위에서 동작합니다. KIST 시스템에 붙이는 방법은 [docs/KIST_통합_가이드.md](docs/KIST_통합_가이드.md)에 따로 정리했습니다.

## 보고서 항목과 폴더

| 보고서 항목 (3차년도) | 폴더 | 파이썬 패키지 | 담당 |
|---|---|---|---|
| **가. 단말/계층형 연합 학습 실험을 위한 시뮬레이션 환경 구축** | `1_simulation/` | | |
| (1) Flask 기반 클라이언트–엣지–전역 서버 3계층 연합학습 | [1-1_hierarchical_fl](1_simulation/1-1_hierarchical_fl/README.md) | `hierarchical_fl` | ghk71 |
| (2) 계층적 연합학습 및 매칭 환경 평가 시각화 | [1-2_fl_visualization](1_simulation/1-2_fl_visualization/README.md) | `defai_addons.monitor` | Soobin |
| (3) 보안 시뮬레이션 환경 고도화 | [1-3_security](1_simulation/1-3_security/README.md) | `security_sim` | ghk71 |
| (4) 지역 분산 컴퓨팅과 연합 컴퓨팅 비교 | [1-4_distributed_comparison](1_simulation/1-4_distributed_comparison/README.md) | `defai_addons.compare` | Soobin |
| (5) 연합학습 시나리오 테스트 | [1-5_scenario_test](1_simulation/1-5_scenario_test/README.md) | `scenario_test` | ghk71 |
| (2차년도) 다계층 네트워크 비용 최적화 | [1-6_network_cost](1_simulation/1-6_network_cost/README.md) | `network_cost` | ghk71 |
| **나. 분석 속도 향상을 위한 다계층 분산 연합 분석 기술** | `2_analysis_speed/` | | |
| (1) 연산 그래프 기반 가지치기와 임계값 동적 조정 | [2-1_graph_pruning](2_analysis_speed/2-1_graph_pruning/README.md) | `graph_pruning` | ghk71 |
| (2) 교사–학생 구조와 중간층 매칭 지식 증류 | [2-2_knowledge_distillation](2_analysis_speed/2-2_knowledge_distillation/README.md) | `layer_kd` | ghk71 |
| (3) MapReduce 기반 다계층 연합 분석 | [2-3_mapreduce_analysis](2_analysis_speed/2-3_mapreduce_analysis/README.md) | `defai_research` | skang132 |
| (4) 그래프 기반 동적 클러스터링 중복 제거 | [2-4_graph_deduplication](2_analysis_speed/2-4_graph_deduplication/README.md) | `defai_research.federated` | skang132 |
| **다. 단말/기관 간 불균일 데이터를 고려한 학습 모델 정확도 향상 기술** | `3_heterogeneous_data/` | | |
| (1) 트랜스포머 기반 멀티모달 학습 모델 | [3-1_multimodal](3_heterogeneous_data/3-1_multimodal/README.md) | `defai_addons.multimodal` | Soobin |
| (2) 연합학습 알고리즘 성능 평가 지표 | [3-2_fl_metrics](3_heterogeneous_data/3-2_fl_metrics/README.md) | `defai_research.metrics` | skang132 |
| (1차년도) 데이터 불균형 해결 | [3-3_data_imbalance](3_heterogeneous_data/3-3_data_imbalance/README.md) | `data_imbalance` | ghk71 |

3차년도 보고서 항목에 없는 1-6과 3-3은 이전 차년도 개발 항목입니다. 후속 항목으로 대체되지 않아 성격이 가까운 장 아래에 두었습니다. 1~3차년도 개발 항목이 어디로 통합됐는지는 [docs/개발항목_통합정리.md](docs/개발항목_통합정리.md)에 있습니다.

각 항목 폴더는 같은 모양입니다. `README.md`에 설명과 실행 방법이 있고, 소스는 `src/` 아래, 실행 스크립트는 `scripts/`나 `examples/`, 이전 실험 기록은 `evidence/`에 있습니다.

```text
Federated_Learning/
├── 1_simulation/            가. 시뮬레이션 환경 (6개 항목)
├── 2_analysis_speed/        나. 분석 속도 향상 (4개 항목)
├── 3_heterogeneous_data/    다. 불균일 데이터 (3개 항목)
├── common/                  여러 항목이 함께 쓰는 코드 (fl_core, defai_addons 공통부)
├── examples/                KIST 시스템 결과를 읽는 예제
├── scripts/                 데모 실행, KIST 연동 점검, 소스 묶음 생성
├── tests/                   modules/ · addons/ · research/ 테스트 (93개)
└── docs/                    KIST 통합 가이드, 개발 항목 정리, 지표 설명, 이전 검증 기록
```

## 설치

Python 3.11~3.13에서 동작합니다. 검증은 Windows 11, Python 3.12.7, CPU 환경에서 했습니다.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -e ".[kist,test]"
```

소스 폴더 이름과 패키지 이름이 다르기 때문에 저장소 루트에서 한 번 설치(`-e`)해야 합니다. 설치하면 모든 패키지와 두 개의 명령(`defai-research`, `defai-addons`)이 함께 등록됩니다.

KIST 시스템과 함께 쓰려면 같은 환경에 KIST `defai`도 설치합니다. 자세한 순서는 [통합 가이드](docs/KIST_통합_가이드.md#1-설치)에 있습니다.

## 바로 확인하기

```bash
.venv/Scripts/python -m pytest
.venv/Scripts/python scripts/run_demo.py
.venv/Scripts/python scripts/check_kist_integration.py
.venv/Scripts/python -m hierarchical_fl dashboard --results results/demo
```

- `pytest`: 세 팀의 테스트 93개를 실행합니다(약 2분 30초). KIST `defai`가 없으면 KIST 연동 테스트 4개는 건너뜁니다.
- `run_demo.py`: 1-1, 1-3, 1-5, 1-6, 2-1, 2-2, 3-3 항목의 실행 명령을 짧은 설정으로 차례로 돌리고 결과를 `results/demo/`에 저장합니다(약 30분).
- `check_kist_integration.py`: KIST `defai`가 설치된 환경에서 연결 방식 6가지를 실제로 실행해 봅니다(약 1분).
- 대시보드: `http://127.0.0.1:5000`에서 계층 구조, 매칭 행렬, 통신 상태, 정확도, 보안 이벤트를 볼 수 있습니다.

1-2, 1-4, 3-1 항목은 `defai-addons` 명령으로, 2-3, 2-4, 3-2 항목은 `defai-research` 명령으로 실행합니다. 각 항목 README에 명령이 있습니다.

## 패키지 이름

| 패키지 | 내용 |
|---|---|
| `fl_core` | Flower Grid·전략 래퍼, 플러그인 인터페이스, 내장/KIST 백엔드, KIST용 ServerApp·ClientApp |
| `hierarchical_fl`, `security_sim`, `scenario_test`, `network_cost`, `graph_pruning`, `layer_kd`, `data_imbalance` | 1-1, 1-3, 1-5, 1-6, 2-1, 2-2, 3-3 |
| `defai_addons` | 1-2, 1-4, 3-1과 공통부 (`common/src/defai_addons`) |
| `defai_research` | 2-3, 2-4, 3-2 |

`defai_research`는 원래 `defai`라는 이름이었습니다. KIST 시스템의 패키지 이름도 `defai`이고 `aggregation.py`, `compression` 같은 같은 이름의 모듈이 있어서, 두 코드를 한 환경에 설치할 수 있도록 이름을 바꿨습니다. 명령 이름도 `defai`에서 `defai-research`로 바뀌었습니다.

## 문서

| 문서 | 내용 |
|---|---|
| [docs/KIST_통합_가이드.md](docs/KIST_통합_가이드.md) | KIST 시스템에 붙이는 방법, 연결 지점, 점검 절차 |
| [docs/개발항목_통합정리.md](docs/개발항목_통합정리.md) | 1~3차년도 개발 항목을 어떤 폴더로 통합했는지 |
| [docs/metrics.md](docs/metrics.md) | 3-2 평가 지표 16개의 정의와 실험 순서 |
| [docs/history/](docs/history/) | 정리 이전에 각 팀이 남긴 범위 문서와 검증 기록 |
| [docs/evidence/](docs/evidence/) | 1-2, 1-4, 3-1 항목의 이전 설치·통합 검증 로그 |

## 라이선스

MIT License ([LICENSE](LICENSE))
