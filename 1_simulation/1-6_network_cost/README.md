# 1-6 다계층 네트워크 비용 최적화

보고서 위치: 2차년도 개발 항목 (3차년도 보고서 가. 시뮬레이션 환경과 연계) · 패키지: `network_cost` · 소스: [`src/network_cost`](src/network_cost) · 테스트: [`tests/modules/test_network_cost.py`](../../tests/modules/test_network_cost.py)

클라이언트–접속 노드(AP)–지역 라우터–엣지–클라우드로 구성된 다계층 DAG 토폴로지 위에서 연합학습의 실제 Flower 메시지(모델 배포·업데이트 업로드)를 전송한다고 보고, 최적화 기술 적용 전후의 데이터 전송량과 네트워크 비용을 라운드별로 비교합니다.
ClientApp 미들웨어(전송 계산·압축)와 Strategy 래퍼(비용 수집)로 구현되어 KIST 시스템과 계층형 시뮬레이터 어디에나 붙습니다.

## 개발 항목 (2차년도)

| 최적화 | 제안서 | 구현 (`optimizations`) |
|---|---|---|
| 엣지 컴퓨팅 | 로컬 노드 전처리·경량화로 업링크 데이터 최소화 | `edge_compute`: 업데이트(로컬−전역)를 상위 k%(`ratio`) 희소화 + 8비트 양자화, 오차 피드백. 압축·복원된 업데이트가 실제 집계에 사용되어 정확도 영향까지 측정 |
| 캐싱 | LRU 캐시로 빈번한 데이터/모델 업데이트 중복 전송 방지 | `cache`: 모델 배포를 64KB 청크(SHA-1)로 나누고 각 하위 노드의 LRU 캐시에 있는 청크는 다시 보내지 않음(같은 AP 아래 다수 클라이언트, 라운드 간 변하지 않은 청크) |
| DAG 라우팅 | 전송 경로를 DAG로 모델링, 비용 함수 기반 최적 경로 선택 | `dag`: 링크 비용 = 가격·MB + (지연 + 전송시간) + 혼잡도(라운드 내 누적 부하), 다익스트라로 최소 비용 경로. 기준은 고정 기본 경로 |

같은 실행에서 기준(최적화 없음, 고정 경로) 비용을 동시에 계산하므로 `*_base`와 `*_opt`를 직접 비교할 수 있습니다.

## 실행

```bash
python -m network_cost --clients 10 --rounds 5
python -m network_cost --variants baseline,all --mode hier --clients 9 --edges 3
python -m network_cost --backend kist --ratio 0.25
```

Python API:

```python
from network_cost.plugin import NetworkCostPlugin
plugin = NetworkCostPlugin({"optimizations": ["edge_compute", "cache", "dag"], "ratio": 0.25, "cache-mb": 64})
```

계층형 시뮬레이터에서는 각 엣지 세션이 클라이언트↔엣지 구간을 계산해 보고하고 전역 서버가 합산합니다. 메시지별 비용은 응답 MetricRecord(`net-*`)로 전달되므로 Flower 배포(`fl_core.kist_apps`)처럼 클라이언트와 서버가 다른 프로세스여도 집계됩니다.

## KIST 시스템과 연결

메시지별 비용은 ClientApp 응답의 MetricRecord(`net-*`)로 전달되어 서버 쪽 `CostCollector`가 합산합니다. 그래서 KIST의 `flwr run` 배포처럼 클라이언트와 서버가 다른 프로세스여도 동작합니다.

## 출력

`results/network_cost/network_cost.json`(변형별 라운드 기록), `network_cost.csv`(정확도, 전송량, 비용, 절감률, 캐시 적중률), `network_cost.png`(라운드별 전송량·비용).

## 측정 결과

측정 환경: `scripts/run_demo.py` 기본 설정(synthetic 데이터, 3라운드 내외의 짧은 실행, CPU i9-11900F). 짧은 실행이므로 절대값보다 같은 설정 안의 비교로 해석합니다.

`python -m network_cost --rounds 3 --clients 8` (모델 6.3 MB, 업데이트 상위 25% + 8비트)

| 변형 | 최종 정확도 | 전송량 (모든 홉) | 절감 | 네트워크 비용 | 절감 | 캐시 적중률 |
|---|---|---|---|---|---|---|
| baseline | 0.6125 | 1218.3 MB | 0% | 348.2 | 0% | - |
| edge_compute | 0.5935 | 796.9 MB | 34.6% | 242.7 | 30.3% | - |
| cache | 0.6125 | 875.6 MB | 28.1% | 300.5 | 13.7% | 0.56 |
| dag | 0.6145 | 1129.5 MB | 7.3% | 229.2 | 34.2% | - |
| all | 0.6025 | 447.8 MB | 63.2% | 125.9 | 63.8% | 0.51 |

캐시와 DAG 경로는 학습 결과를 바꾸지 않고(정확도 동일), 엣지 압축은 전송량을 줄이는 대신 정확도가 약간 낮아집니다.
