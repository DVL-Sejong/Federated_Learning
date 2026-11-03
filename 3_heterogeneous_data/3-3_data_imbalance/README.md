# 3-3 데이터 불균형 해결

보고서 위치: 1차년도 개발 항목 (3차년도 보고서 다. 불균일 데이터와 연계) · 패키지: `data_imbalance` · 소스: [`src/data_imbalance`](src/data_imbalance) · 테스트: [`tests/modules/test_data_imbalance.py`](../../tests/modules/test_data_imbalance.py)

클라이언트의 클래스 분포 불균형은 조합 샘플링으로, 클라이언트 간 데이터 양 불균형으로 인한 전역 모델 편향은 데이터량·로컬 손실 기반 동적 가중치로 완화합니다.

## 개발 항목 (1차년도 설계 → 구현)

| 설계 항목 | 구현 |
|---|---|
| 각 클라이언트의 불균형 데이터 기반 클래스 분포 조정 | `resample(x, y, method)`를 ClientApp 데이터 변환 훅으로 적용 (내장·KIST 데이터 모두) |
| SMOTE로 소수 클래스 증폭 | `smote`: 같은 클래스 k-최근접 이웃 사이 보간으로 최다 클래스 수(× `target-ratio`)까지 합성 |
| Tomek Links, ENN으로 노이즈 제거·정제 | `tomek_links`(상호 최근접·다른 클래스 쌍의 다수 클래스 표본 제거), `enn`(k-NN 다수결 불일치 표본 제거, 소수 클래스 보호) |
| 오버샘플링 + 언더샘플링 조합 | `combo` = SMOTE → Tomek → ENN (비교: `smote`, `tomek`, `enn`, `smote_tomek`, `smote_enn`) |
| 조합 샘플링 효과 평가용 분류 모델 | 로컬 단독 학습 비교(`--local-study`)와 연합학습 비교, macro-F1·최소 클래스 재현율 |
| 데이터 양·로컬 손실 기반 동적 가중치로 글로벌 업데이트 | `DynamicWeightStrategy`: `w_i ∝ n_i^α · (L_i / mean L)^β`, L_i = 수신한 전역 모델의 로컬 손실(EMA 평활). 임의 Flower Strategy의 가중치 키(`num-examples`)를 바꿔 집계 |
| 가중치 적용 전후 비교 | `fedavg`(n_i 가중) / `weighting` / `sampling` / `sampling+weighting` |

## 실행

```bash
python -m data_imbalance --clients 10 --rounds 5 --alpha-dirichlet 0.3 --local-study
python -m data_imbalance --backend kist --variants fedavg,weighting
```

Python API:

```python
from data_imbalance.plugin import ImbalancePlugin
from fl_core.plugins import run_flat
from fl_core.backends import get_backend

plugin = ImbalancePlugin({"sampling": "combo", "weighting": "size_loss", "alpha": 1.0, "beta": 1.0})
out = run_flat(get_backend("builtin", {"partitioner": "dirichlet", "dirichlet-alpha": 0.3}), [plugin], 10, 5)
```

계층형 시뮬레이터에서는 엣지·전역 계층 모두에 동적 가중치를 적용합니다(`tiers`). 샘플링을 쓰면 KIST 백엔드에서도 KIST 데이터·모델로 학습하는 플러그인 학습 핸들러가 사용되고, 가중치만 쓰면 KIST ClientApp 그대로 동작합니다.

## KIST 시스템과 연결

동적 가중치(`DynamicWeightStrategy`)는 KIST 전략의 가중치 키(`num-examples`)만 바꾸므로 KIST ClientApp을 수정하지 않고 붙습니다. 조합 샘플링은 KIST 데이터·모델로 학습하는 플러그인 학습 핸들러를 사용합니다.

## 출력

`results/data_imbalance/imbalance.json`(변형별 정확도 곡선, macro-F1, 클래스별 재현율, 라운드 가중치 기록, 로컬 연구 결과), `imbalance.csv`.

## 측정 결과

측정 환경: `scripts/run_demo.py` 기본 설정(synthetic 데이터, 3라운드 내외의 짧은 실행, CPU i9-11900F). 짧은 실행이므로 절대값보다 같은 설정 안의 비교로 해석합니다.

`python -m data_imbalance --rounds 3 --clients 8 --local-study` (Dirichlet α=0.3)

| 변형 | 최종 정확도 | macro-F1 | 최소 클래스 재현율 |
|---|---|---|---|
| fedavg (데이터량 가중) | 0.622 | 0.570 | 0.000 |
| sampling (SMOTE+Tomek+ENN) | 0.856 | 0.851 | 0.525 |
| weighting (데이터량·손실) | 0.663 | 0.623 | 0.000 |
| sampling+weighting | 0.788 | 0.778 | 0.480 |

로컬 단독 학습: 클라이언트 평균 불균형 비율 131.0 → 2.1, 로컬 검증 정확도 0.581 → 0.491, 전역 테스트 정확도 0.271 → 0.489. 1차 보고서의 관찰(로컬 모델은 차이가 작거나 낮아지고 전역 모델은 조합 샘플링에서 가장 우수)과 같은 경향입니다.
4라운드 실행에서는 fedavg 0.815 / sampling 0.902 / weighting 0.838 / sampling+weighting 0.903으로, 동적 가중치 효과는 라운드 수와 분할에 따라 달라집니다.
