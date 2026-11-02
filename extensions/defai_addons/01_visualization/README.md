# 계층형 연합학습 평가 시각화 모듈

1차년도 성능 모니터링 설계, 2차년도 시뮬레이션 모니터링, 3차년도 매칭 및 알고리즘 평가를 하나의 모듈로 통합했습니다. 주요 코드는 `defai_addons/monitor`에 있습니다.

- 매칭 학습 추세와 업데이트 이상치 탐지
- 계층 구조 tree, 우선순위 rank, 서버 통신 상태 및 매칭 행렬
- FedAvg, FedProx, q-FedAvg의 학습 추세와 성능 비교
- 데이터 분포 및 엔트로피, 클라이언트 유사도, 통신 흐름 및 자원 사용량

## 설치 및 독립 실행

상위 폴더에서 `python -m pip install ".[dashboard]"`를 실행합니다.

```powershell
defai-addons dashboard --port 5055 --store C:/experiments/addon-monitor
```

브라우저에서 `http://127.0.0.1:5055`를 엽니다. 소스 작업 공간에서는 `python run.py --port 5055`로도 실행할 수 있습니다. MNIST를 처음 사용할 때는 다운로드가 필요합니다. 데이터 캐시는 Git에 포함하지 않습니다. 환경변수 `DEFAI_DATA`로 쓰기 가능한 캐시 폴더를 지정할 수 있습니다.

## 기존 GitHub 시스템의 결과 호출

```powershell
defai-addons visualize C:/experiments/fedavg/history.json C:/experiments/fedprox/history.json C:/experiments/qfedavg/history.json --out C:/experiments/algorithm-view.html
```

```python
from defai_addons.api import read_history, render_histories

record = read_history("C:/experiments/fedavg/history.json")
render_histories(["C:/experiments/fedavg/history.json"],
                 "C:/experiments/fedavg-view.html")
print(record["records"][-1]["global_accuracy"])
```

HTML에는 라운드별 글로벌 정확도, 클라이언트 표본 수 가중 정확도, 누적 통신량, 최종 결과 표, 입력 설정 및 원본 해시가 포함됩니다. `eval_acc`를 로컬 F1로 바꾸지 않습니다. 단위는 MiB이며 in-process 통신량은 실제 네트워크 측정값이 아닙니다.

기존 시스템 파일에는 매칭 및 계층 정보가 없으므로 해당 결과로 가상의 tree나 매칭 행렬을 그리지 않습니다. 기존 계층 시각화는 모듈 자체 시뮬레이터 또는 그 스키마에 맞는 외부 이벤트를 사용합니다.

## 외부 서비스에서 대시보드 호출

```python
from defai_addons.api import create_dashboard

app = create_dashboard(store_dir="C:/experiments/addon-monitor")
app.run(host="127.0.0.1", port=5055)
```

| API | 용도 |
|---|---|
| `GET /api/defaults` | 시뮬레이션 및 비교 설정 조회 |
| `POST /api/runs` | `{"config":{"algorithm":"fedavg","rounds":3}}`로 모듈 시뮬레이션 실행 |
| `GET /api/runs/<id>` | 진행 상태 및 계산된 시각화 데이터 조회 |
| `POST /api/ext/runs` | `config`, `topology`, `clients`로 외부 실행 등록 |
| `POST /api/ext/runs/<id>/rounds` | 모듈 라운드 스키마로 이벤트 전달 |
| `POST /api/ext/runs/<id>/finish` | 외부 실행 종료 |

외부 이벤트의 실제 예제는 `scripts/ext_client_example.py`와 `defai_addons/monitor/deploy.py`에 있습니다. `MonitoringStrategy`는 기존 Flower Strategy API용으로, GitHub 시스템의 새 ServerApp 전략에 그대로 끼우는 어댑터가 아닙니다. GitHub 시스템을 변경하지 않을 때는 완료된 결과 파일 연결을 사용합니다.

## 검증 및 출력

상위 폴더의 `integration_tests.log`는 결과 변환, 누락 지표 처리, 원본 불변 검사를 포함합니다. `verification_layout.json`에는 Flask 페이지 및 API 검사가 있습니다. 작업 공간의 `runs`와 `results`에 자체 실행 기록을 저장합니다. 네트워크 서비스로 공개 배포할 경우 별도 운영 서버와 접근 제어 구성이 필요합니다.
