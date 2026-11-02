# 지역 분산 컴퓨팅과 연합 컴퓨팅 비교 모듈

1차년도 시뮬레이션 환경 설계를 3차년도 비교 환경 및 평가 절차로 통합했습니다. 주요 코드는 `defai_addons/compare`에 있습니다.

- 지역 분산: 사이트 독립 학습, 데이터 재분할 후 동기화, 중앙 집중 학습
- 연합: Flower 기반 FedAvg 및 FedProx, 쌍별 마스크 기반 집계, AES-GCM 업데이트 보호 및 차분 프라이버시 비교
- 정제, 정규화, PCA 전처리와 글로벌 및 로컬 정확도, 정밀도, 재현율, Macro F1 계산
- 원본 데이터 이동량, 모델 및 통계 통신량, F1 차이의 부트스트랩 신뢰구간

보안 및 통신 검사는 시뮬레이션 범위입니다. 이 모듈 실행으로 기존 GitHub 시스템의 전송 경로나 보안 설정이 변경되지는 않습니다. 같은 프로세스에서 사이트를 모사하며 독립 서버 간 운영 배포를 자동 구성하지 않습니다.

## 설치 및 CLI 호출

상위 폴더에서 `python -m pip install ".[learning]"`를 실행합니다. `examples/comparison_config.json`은 작은 동작 확인용 설정입니다.

```powershell
defai-addons compare --config examples/comparison_config.json --out C:/experiments/comparison.json
```

상위 폴더에서 실행하는 예시입니다. 자체 MNIST 실험은 이 폴더의 `python run.py --help`로 옵션을 확인할 수 있습니다. 작은 동작 확인용 실험 결과는 연구 성능 수치로 사용하지 않습니다.

## 기존 시스템의 데이터로 Python 호출

기존 저장소 밖의 연결 스크립트에서 사이트별 배열을 전달합니다. 입력 배열은 복사하므로 호출자의 데이터를 변경하지 않습니다.

```python
import numpy as np
from defai_addons.api import compare_arrays

# caller_sites는 호출자가 정한 사이트별 학습 및 로컬 평가 데이터입니다.
sites = [{"X": s["X_train"], "y": s["y_train"],
          "X_test": s["X_test"], "y_test": s["y_test"]}
         for s in caller_sites]
report = compare_arrays(sites, global_X_test, global_y_test, config={
    "epochs": 10, "local_epochs": 1, "use_pca": True,
    "encrypt_updates": True, "dp_noise": 0.3, "dp_clip": 2.0,
    "envs": ["ld_independent", "ld_sync", "ld_central",
             "fl_fedavg", "fl_fedavg_dp"],
})
print(report["results"][0]["global"]["f1"])
```

`X`와 `X_test`는 `[N,D]` 실수 배열, `y`와 `y_test`는 `[N]` 정수 배열입니다. 클래스 번호는 0부터 시작합니다. 모든 사이트의 특징 차원은 같아야 합니다. 이미지 입력은 호출자가 동일한 기준으로 펼치거나 특징을 추출해 전달해야 합니다. 비교 엔진은 MLP를 사용하며 기존 시스템의 CNN 학습 결과를 그대로 재현하는 API가 아닙니다.

위의 `caller_sites`, `global_X_test`, `global_y_test`는 호출자가 준비하는 데이터 변수입니다. 기본 내장 데이터로 실행하려면 `from defai_addons.api import compare; report = compare(config)`를 사용합니다.

## 별도 프로세스에서 NPZ 호출

```python
import numpy as np
payload = {"n_sites": len(sites), "X_test": global_X_test, "y_test": global_y_test}
for i, site in enumerate(sites):
    payload.update({f"site_{i}_{key}": value for key, value in site.items()})
np.savez("C:/experiments/sites.npz", **payload)
```

```powershell
defai-addons compare --config examples/comparison_config.json --data C:/experiments/sites.npz --out C:/experiments/site-comparison.json
```

파일로 전달하는 방식은 로컬 평가 도구로 데이터를 모으는 작업입니다. 이를 실제 분산 운영에서 원본 데이터가 이동하지 않는 방식으로 설명하면 안 됩니다. 실제 연합 원본 비전송 속성은 모듈이 모사하는 학습 프로토콜에 해당합니다.

반환 JSON에는 `results[].global`, `local`, `local_mean`, `cost`, `privacy`, `history`, `f1_diff_vs_ref`가 있습니다. 서버에 학습 데이터가 전달되는지 여부는 `privacy.raw_data_leaves_site`로 구분합니다.

## 기존 GitHub 결과와의 관계

기존 `history.json`에는 F1과 원시 예측값이 없어 글로벌 및 로컬 F1을 역산할 수 없습니다. 기존 결과의 정확도 및 통신량 조회는 `defai_addons.api.read_history`로 연결합니다. F1 비교가 필요하면 위 API에 평가 데이터를 전달하여 별도 실험을 실행합니다. 모델 및 분할 설정이 다른 수치를 같은 조건의 성능 비교로 표시하지 않습니다.

검증 기록은 상위 폴더의 `integration_tests.log`에 있습니다. 외부 배열을 실제 비교 엔진에 전달하고 지역 분산 및 연합 학습 실행, F1 계산, 입력 불변 여부를 확인합니다.
