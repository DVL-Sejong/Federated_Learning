# deFAI 확장 모듈

이 저장소의 기존 연구 모듈을 변경하지 않고 별도 폴더에 추가한 시각화, 지역 분산 비교 및 멀티모달 학습 확장 모듈입니다.

기존 [rixile9999/deFAI](https://github.com/rixile9999/deFAI) 시스템의 결과를 읽는 어댑터도 포함합니다. 이 외부 시스템의 연동 기준은 커밋 `b21d628b8f5042a00d46da5a2280987c6469f281`입니다. 아래의 `defai run`, `defai.simulation`, `history.json` 예시는 해당 외부 시스템용이며, 현재 저장소의 기존 CLI 또는 API에 같은 기능이 있다는 의미가 아닙니다. 세 확장 모듈 자체는 `defai-addons` 명령과 `defai_addons.api`로 독립 호출할 수 있습니다.

| 작업 공간 | 기능 | 호출 설명 |
|---|---|---|
| `01_visualization` | 계층형 연합학습 평가 시각화 및 기존 시스템 결과 시각화 | [README](01_visualization/README.md) |
| `02_comparison` | 지역 분산 컴퓨팅과 연합 컴퓨팅 비교 | [README](02_comparison/README.md) |
| `03_multimodal` | Transformer 기반 멀티모달 학습 및 추론 | [README](03_multimodal/README.md) |
| `shared` | 공통 연산, 공개 API, 기존 시스템 결과 어댑터 | `shared/defai_addons/api.py` |

항목별 소스, 실행 스크립트, 결과, VS Code 작업 공간을 분리했습니다. 공통 코드는 한 곳에서 관리하므로 전체 폴더를 함께 보관합니다. 1~3차년도 통합 기준과 PDF 대응은 `통합_개발항목.txt` 및 `연차별_통합대응표.csv`에 있습니다. 이전 완성 모델은 기본 실행 대상에서 제외하고 `baselines`에 재현용으로 보존했습니다.

## 설치

저장소 루트에서 `cd extensions/defai_addons`로 이동한 뒤 이 폴더에서 실행합니다. 기존 시스템과 같은 Python 환경에서 사용할 경우 **Python 3.11~3.13**을 사용하고 기존 시스템의 설치 지침과 버전 제약을 우선 적용합니다.

```powershell
python -m pip install ".[dashboard]"
defai-addons --help
```

결과 파일 연동만 필요한 환경에는 학습 라이브러리 없이 설치할 수 있습니다.

```powershell
python -m pip install .
```

패키지 이름은 `defai-addons`, Python 이름은 `defai_addons`, 명령은 `defai-addons`입니다. 기존 `defai` 패키지와 명령을 덮어쓰지 않습니다. wheel은 저장소의 기존 제외 규칙에 따라 커밋하지 않으며, 필요하면 다음과 같이 생성한 후 설치할 수 있습니다.

```powershell
python -m pip wheel . --no-deps --wheel-dir dist
python -m pip install "dist/defai_addons-3.2.0-py3-none-any.whl[dashboard]"
```

의존성 충돌을 피하려면 별도 가상환경에 확장 모듈을 설치하고 파일 또는 별도 프로세스로 호출하는 방식을 권장합니다. 이 방식에서는 기존 시스템 환경을 변경할 필요가 없습니다. `requirements.txt`는 기존 실험 재현용이며 배포 의존성은 `pyproject.toml`을 기준으로 합니다.

## 기존 시스템에서 호출

### 1. 기존 실행 결과 연결: 시스템 코드 수정 불필요

기존 시스템을 평소처럼 실행한 뒤 생성된 `history.json`을 전달합니다. 아래 경로는 사용자의 실제 위치에 맞게 바꿉니다.

```powershell
defai run --strategy fedavg --rounds 3 --clients 4 --dataset synthetic --device cpu --output C:/experiments/host-run
defai-addons visualize C:/experiments/host-run/history.json --out C:/experiments/addon-view.html
defai-addons history C:/experiments/host-run/history.json --out C:/experiments/addon-history.json
```

확장 CLI가 다른 가상환경에 있으면 그 환경의 `Scripts/defai-addons.exe`를 절대 경로로 호출합니다. 입력 파일은 읽기 전용이며 출력 경로가 이미 존재하면 덮어쓰지 않고 종료합니다.

### 2. 외부 연결 스크립트에서 Python 호출

기존 저장소 밖에 별도의 스크립트를 두면 됩니다. 두 패키지를 설치한 환경에서는 다음처럼 함께 사용할 수 있습니다.

```python
from defai.simulation import run_simulation
from defai_addons.api import render_histories

run_simulation("C:/experiments/host-python-run", clients=4,
               overrides={"dataset": "synthetic", "num-server-rounds": 3})
render_histories(["C:/experiments/host-python-run/history.json"],
                 "C:/experiments/host-python-view.html")
```

데이터를 직접 전달하는 비교 API와 멀티모달 모델 API는 각 README에 있습니다. 기존 시스템의 CNN 및 Flower ServerApp에 모델을 자동 등록하는 기능은 없습니다. 멀티모달 학습은 독립 호출하거나 별도의 클라이언트에서 명시적으로 사용합니다.

같은 환경에서 사용할 연결 스크립트는 `examples/run_host_and_visualize.py`에 있습니다.

```powershell
python examples/run_host_and_visualize.py --host-output C:/experiments/host-connected --view-output C:/experiments/host-connected.html
```

소스를 수정한 뒤 설치 패키지로 실행할 때는 `python -m pip install ".[dashboard]"`를 다시 실행해 수정 사항을 반영합니다. 각 작업 공간의 `run.py`는 해당 소스를 직접 사용합니다.

## 연결 범위와 검증

- 기존 시스템의 결과에 있는 글로벌 정확도, 클라이언트 가중 정확도, 손실, 누적 학습 통신량을 읽습니다.
- 기존 `history.json`에 없는 F1, 정밀도, 재현율, 계층 구조 및 매칭 정보는 생성하지 않습니다. 비교 모듈이 직접 실행한 실험에서는 해당 평가 지표를 계산합니다.
- 기존 시스템은 실행 종료 후 파일을 저장하므로 파일 연결은 완료된 실행의 분석입니다. 실시간 모니터링은 별도의 라운드 이벤트 연동이 필요합니다.
- `integration_tests.log`, `installation_verification.json`, `verification_layout.json`에 검증 결과가 있습니다. 기존 시스템 전체 학습 및 Flower 1.42 배포와의 통합 실행은 이번 환경에서 검증하지 않았습니다.
- 기존 시스템 파일은 변경하지 않습니다. 확장 모듈은 `extensions/defai_addons/` 아래에만 추가합니다.

```powershell
python -m pip install ".[dashboard,test]"
python -m pytest tests -q
python tests/check_workspace_layout.py
node tests/test_dashboard_metrics.js
```
