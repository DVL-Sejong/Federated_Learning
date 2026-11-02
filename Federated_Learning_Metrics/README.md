# Federated Learning Metrics

연합학습 알고리즘 성능 평가 지표 모듈

16개 지표의 정의, 단위와 계산 결과를 공통 API로 제공하고, 실제 센서와 학습 이력 및 전송 기록을 이용하는 목적별 검증 실험과 연결
정밀도와 재현율 계산은 보고서 생성 코드와 분리하여 HAR의 6개 활동과 Digits의 10개 클래스를 같은 API로 평가

## 평가 항목

- 정밀도, 재현율과 F1
- 수렴 시간과 통신 효율
- 멀티모달 균일성, 적응력과 지식 통합도
- CPU 시간과 메모리 사용 및 프라이버시
- 데이터 엔트로피와 공동 정보
- 전송 정보 손실, 정보 복원과 정보 전송률
- 노드 간 정보 활용

16개 항목별 입력, 단위와 검증 실험은 [지표 설명](../docs/metrics.md) 참조
예측만으로 산출하는 지표와 별도 대조 실험이나 송수신 기록이 필요한 지표를 구분
관측하지 않은 항목은 `not_measured`로 유지하고, 정의되지 않는 분모는 수치로 임의 대체하지 않음

## 소스 위치

- [api.py](src/defai/metrics/api.py): 16개 지표 정의와 공통 계산 및 결과 구성
- [studies/purpose_multimodal_study.py](src/defai/metrics/studies/purpose_multimodal_study.py): 실제 센서 조합, 입력 중단 및 복귀, 로컬 모델과 전역 모델 비교
- [studies/purpose_system_study.py](src/defai/metrics/studies/purpose_system_study.py): 실제 송수신, 수렴과 전송 비용 및 CPU와 메모리 측정
- [studies/purpose_recovery_stress.py](src/defai/metrics/studies/purpose_recovery_stress.py): 업데이트 손상과 복원 대조
- [studies/purpose_privacy_study.py](src/defai/metrics/studies/purpose_privacy_study.py): 실제 클리핑과 Gaussian 잡음, 프라이버시 예산 및 회원추론 공격 대조
- `studies/verify_purpose_*.py`: 보존한 원시 자료와 전송 기록에 대한 독립 재계산

주요 API는 `metric_definitions`, `classification_metrics`와 `evaluate16`
`evaluate16`은 라벨 및 예측 확률과 지표별 실제 관측을 받아 16개 항목의 이름, 목적, 단위, 값과 측정 상태 반환
공통 Flower 클라이언트는 [그래프 모듈의 학습 도구](../Graph_Dynamic_Deduplication/src/defai/federated/learning.py)를 재사용

## 설치와 실행

이 모듈 폴더 안에서 시작하는 경우 상위 `Federated_Learning`으로 이동하여 세 모듈을 한 번 설치
실행 확인 환경은 Python 3.12

```sh
cd ..
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[transformer,plots]"
defai metrics-schema
defai metrics --input predictions.npz --observations observations.json --output results/metrics.json
```

예측 NPZ의 필드는 `labels`와 `probabilities`이며 추가 관측 JSON의 키는 지표 정의의 `field` 값 사용
목적별 실험은 아래 명령으로 시작하며 세부 순서와 필요한 추가 검증은 [지표 설명](../docs/metrics.md) 참조

```sh
defai metric-study multimodal -- --download --prepare-only
defai metric-study multimodal
defai metric-study system -- --stage smoke --output results/system_smoke
defai metric-study privacy
```

HAR 다운로드는 `--download`를 명시한 경우에만 수행
데이터와 모델 및 실험 결과는 공개 소스와 분리하여 저장
프라이버시 예산은 실제 적용한 메커니즘과 공개 범위에 대해 해석하고 공격 AUC와 구분

전체 설치 안내는 [상위 README](../README.md), 라이선스는 [MIT License](../LICENSE) 참조
