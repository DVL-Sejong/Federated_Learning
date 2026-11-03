# 2-1 연산 그래프 기반 모델 가지치기

보고서 위치: 3차년도 보고서 나-(1) · 패키지: `graph_pruning` · 소스: [`src/graph_pruning`](src/graph_pruning) · 테스트: [`tests/modules/test_graph_pruning.py`](../../tests/modules/test_graph_pruning.py)

모델을 `torch.fx`로 추적해 연산 그래프를 만들고, 채널 의존성 그룹의 노드·엣지 경로 중요도로 구조적 가지치기를 수행하며, 손실 변화를 감시해 임계값을 동적으로 조정합니다.
중앙(서버) 단독 실행과, 연합학습 중 전역 모델을 가지치기하는 `GraphPruningStrategy`(Flower Strategy 래퍼) 두 방식으로 사용합니다.

## 연차별 개발 항목 통합

| 연차 | 개발 항목 | 이 모듈에서의 처리 |
|---|---|---|
| 1차 | AI 모형 소규모화 설계: 임계값 이하 뉴런·필터 제거, 가중치 절대값 기준 제거 | 필터 단위 제거는 3차 구조적 가지치기로 **대체**. 가중치 절댓값 기준 균일 L1 필터 가지치기는 비교 기준(`uniform_l1`, 파라미터 수를 맞춘 이분 탐색)으로 유지 |
| 2차 | 고도화된 소규모화: BN 스케일 인자 L1 정규화(구조적 희소성), 기울기×활성 기반 중요도, 학습 중 임계값 조정 | BN-L1 희소성 정규화를 로컬 학습 손실 훅(`--bn-l1`, `bn-l1`)으로 **유지**. 기울기 기반 중요도는 1차 테일러 중요도(`--importance taylor`)로 유지. 임계값 조정은 3차 동적 조정으로 **대체** |
| 3차 | 연산 그래프 기반 가지치기: 연산 그래프 생성 → 노드·엣지 경로 중요도 평가 → 임계값 미만 제거, 경험적 초기 임계값 → 손실 변화 감시 → 임계값 동적 조정 | **최신 구현**: 아래 전체 |

## 알고리즘

1. **연산 그래프 생성** (`graph.py`): `torch.fx` 추적 + 형상 전파. Conv/Linear 출력이 채널 공간을 만들고, BN·활성·풀링·flatten은 통과, 잔차 덧셈은 union-find로 공간을 병합(결합 그룹), 입력·출력·그룹 합성곱·공유 모듈은 고정합니다.
2. **경로 중요도**: 채널 c에 대해 들어오는 가중치(생산자)와 나가는 가중치(소비자)의 정규화 제곱합 × BN |γ| 평균, 그룹 평균 1로 정규화한 뒤 가지치기 전 모델의 척도에 고정(anchor)합니다.
3. **가지치기**: 중요도 ≥ τ 채널만 남기고(그룹별 최소 max(4, ⌈0.1·C₀⌉)) Conv/BN/Linear를 물리적으로 재구성, 원래 채널 인덱스(`prune_meta`)를 누적 기록합니다.
4. **임계값 동적 조정** (`dynamic.py`): τ₀ = 중요도 분위수(경험적 초기값) → 가지치기 → 짧은 회복 학습 → 검증 손실 변화 d = (L − L_ref)/L_ref. d ≤ tol이면 수락하고 τ ← τ(1+η), 아니면 직전 모델로 되돌리고 η ← η/2.
5. **연합학습 적용** (`federated.py`): 전역 집계 후 지정 라운드마다 중요도·임계값으로 채널 마스크를 갱신합니다. 이후 라운드의 연합 학습이 회복 학습 역할을 하며, 다음 시점에 손실 변화를 확인해 수락/롤백·η 조정을 합니다. 마스크 방식이라 KIST ClientApp 등 기존 클라이언트가 같은 모델 구조로 학습하고, 종료 후 `final_models()`로 물리적으로 축소한 모델을 얻습니다(마스크 모델과 출력 동일).

## 실행

```bash
python -m graph_pruning graph --set model=resnet8
python -m graph_pruning central --set model=cnn_bn --bn-l1 1e-4 --iters 6
python -m graph_pruning central --importance taylor --target 0.7
python -m graph_pruning federated --clients 8 --rounds 8 --start 2 --every 2
python -m graph_pruning federated --backend kist
```

Python API:

```python
from graph_pruning.federated import GraphPruningPlugin
from fl_core.plugins import run_flat
from fl_core.backends import get_backend

plugin = GraphPruningPlugin({"start": 2, "every": 2, "percentile": 30, "tol": 0.15})
out = run_flat(get_backend("kist", {"dataset": "synthetic"}), [plugin], num_clients=8, rounds=8)
masked, compact = plugin.strategy.final_models(out["result"].arrays)
```

계층형 시뮬레이터에서는 전역 계층(`tiers=["global"]`)에서 동작합니다.

## KIST 시스템과 연결

`GraphPruningStrategy`가 KIST 전략을 감싸 전역 모델에 채널 마스크를 적용합니다. 마스크 방식이라 KIST ClientApp은 원래 `Net` 구조 그대로 학습하고, 학습이 끝난 뒤 `final_models()`로 실제 크기를 줄인 모델을 얻습니다. 검증 손실 증가율(`tol`)과 정확도 하락(`acc-tol`)을 함께 보고 가지치기를 수락하거나 되돌립니다.

## 출력

`results/graph_pruning/`: `central_pruning.json`(반복별 τ·η·제거 채널·파라미터·MACs 감소율·검증 손실·수락 여부, 기준 모델/그래프 가지치기/균일 L1 비교표), `pruned_model.pt`/`.json`(구조 사양 포함 저장), `federated_pruning.json`, `federated_pruned_model.pt`.

## 측정 결과

측정 환경: `scripts/run_demo.py` 기본 설정(synthetic 데이터, 3라운드 내외의 짧은 실행, CPU i9-11900F). 짧은 실행이므로 절대값보다 같은 설정 안의 비교로 해석합니다.

중앙: `python -m graph_pruning central --set model=cnn_bn --bn-l1 1e-4 --iters 4`

| 모델 | 파라미터 | MACs | 지연시간(배치 1) | 테스트 정확도 |
|---|---|---|---|---|
| 기준 (BN-L1 학습) | 1,663,562 | 12.27 M | 1.41 ms | 0.973 |
| 연산 그래프 + 동적 임계값 | 454,880 (−72.7%) | 8.80 M (−28.3%) | 0.92 ms | 0.987 |
| 균일 L1 (같은 파라미터 수) | 444,402 | 3.51 M | 1.16 ms | 0.990 |

동적 조정 기록: τ 0.27 → 0.40 → 0.61 수락, 0.91에서 손실 증가율 0.29 > tol 0.15로 거부 후 되돌림.

연합(KIST 호환 마스크 방식): `python -m graph_pruning federated --rounds 5 --clients 6` — 2라운드에 183채널 가지치기, 4라운드 손실 확인 후 수락(손실 변화 −79%), 최종 축소 모델 103,588 파라미터(−93.8%), 0.92 ms → 0.40 ms, 마스크 모델과 축소 모델의 정확도 0.917로 동일.
