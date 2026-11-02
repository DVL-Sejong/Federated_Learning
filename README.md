# Federated Learning 연구 모듈 v1.0.1

MapReduce 기반 다계층 연합 분석, GraphSAGE 기반 동적 중복 제거, Transformer 압축 전송 및 연합학습의 16개 성능 평가 지표를 통합한 Python 패키지

기존 저장소의 `Federated_Learning` 하위에 추가하는 세 모듈 구성
새 저장소 루트로 사용하는 배포본이 아니며 다른 기존 연구 폴더와 함께 배치
차년도별 복사본을 두지 않고 최신 구현을 기능별로 구성했으며, 기존 기능 중 통계 집계와 독립 검산 및 필요한 비교 조건은 유지
MIT 라이선스 적용

## 폴더 구성

```text
기존 저장소/
└── Federated_Learning/
    ├── MapReduce_Hierarchical_Federated_Analysis/
    │   ├── README.md
    │   ├── src/defai/aggregation.py
    │   ├── src/defai/compression/
    │   └── scripts/verify_transformer_tcp.py
    ├── Graph_Dynamic_Deduplication/
    │   ├── README.md
    │   ├── src/defai/federated/
    │   └── examples/
    ├── Federated_Learning_Metrics/
    │   ├── README.md
    │   └── src/defai/metrics/
    ├── pyproject.toml
    ├── LICENSE
    ├── tests/
    ├── scripts/
    └── docs/
```

- [MapReduce 기반 다계층 연합 분석](MapReduce_Hierarchical_Federated_Analysis/README.md): 블록 집계, 충분통계 집계, Transformer 압축과 수신 복원 및 Flower 계층별 집계
- [그래프 기반 동적 중복 분석 제거](Graph_Dynamic_Deduplication/README.md): GraphSAGE 연관성 학습, 동적 후보 그래프, 동일성 검증, 캐시 및 비교 시각화
- [연합학습 성능 평가 지표](Federated_Learning_Metrics/README.md): 16개 지표 API와 실제 센서, 시스템, 복원 및 프라이버시 검증

세 폴더는 실제 구현 소스를 소유하며, 상위 `pyproject.toml`로 함께 설치하면 하나의 `defai` 패키지로 연결
그래프 폴더의 학습 클라이언트, 데이터 분할과 불변 스냅샷은 다른 모듈에서도 사용하는 공통 기능으로 유지
공통 코드를 폴더마다 복사하지 않으며 독립 집계 검산, 결과 검증과 배포 도구도 한 곳에서 공유
폴더명은 저장소에서 연구 주제별 코드를 찾기 위한 구분이며 세 개의 별도 가상 환경을 요구하지 않음

기존 저장소에 반영할 때는 ZIP의 `Federated_Learning` 안 내용을 해당 폴더에 추가
이 묶음 밖의 기존 저장소 파일과 다른 연구 모듈은 변경 대상에 포함하지 않음
MIT 라이선스의 적용 대상은 이 묶음의 연구 코드

## 설치

`Federated_Learning` 폴더에서 아래 명령 실행

실제 실행을 확인한 환경은 Python 3.12이며 패키지의 지원 범위는 Python 3.11과 3.12
실험의 직접 의존성은 버전을 고정했으며, 기본 학습은 CPU로 수행
기존 Python 3.10 시각화 환경과 별도의 가상 환경을 사용하는 구성

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux: source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[transformer,plots]"
defai --help
python -m unittest discover -s tests -v
```

설치 시 `flwr==1.22.0`을 사용하며 해당 버전의 `NumPyClient`와 `FedAvg` API를 호출
새 Flower 버전으로 변경할 때는 수신 배열과 집계 결과의 검증을 함께 수행
원격 데이터 전송 기능이 필요 없는 기본 벤치마크는 공개 Digits 및 Breast Cancer 자료를 scikit-learn에서 로드
별도 UCI HAR 다운로드는 사용자가 `--download`를 지정할 때만 수행

## 집계 검증

```sh
defai verify-aggregation --output results/aggregation_check.json
defai verify-aggregation --full --output results/aggregation_full.json
```

전체 검증은 110개 벡터 집계 조건과 30개 충분통계 조건을 확인
표본수가 다른 그룹의 가중 집계, 독립 shuffle 집계, 직접 평균 및 분산을 대조
이 검증은 수치 결과에 대한 확인이며 분산 처리 속도의 측정과는 구분

## 압축기 학습과 연합 분석

배포본에 데이터와 학습 가중치를 포함하지 않으므로 처음 실행할 때 학습용 업데이트를 수집하고 압축기를 생성
수집은 고정된 전역 학습 표본만 사용하며, 압축기 내부 검증용 수집 시드는 최적화용 시드와 분리

```sh
defai collect-calibration --output results/calibration
defai train-codec --calibration results/calibration --output results/codec
defai benchmark --dataset digits --artifact results/codec/inference/codec_model.bin --output results/digits_transformer
defai benchmark --dataset digits --codec compact --output results/digits_compact
defai benchmark --dataset digits --codec legacy --output results/digits_legacy_reference
defai benchmark --dataset digits --artifact results/codec/inference/codec_model.bin --static --output results/digits_static
```

기본 설정은 단말 12개, 그룹 4개, 80라운드, 데이터 변경 41라운드, GraphSAGE 학습 100 epoch
Transformer의 기본 구성은 d_model 8, 2개 헤드, 2개 잠재 토큰, INT8 잠재 표현, 상위 8개 INT8 잔차와 블록 이득 보정
수신기는 배포 패킷에서 가중치를 구성하며 수신 복원한 업데이트를 그룹과 중앙 서버의 실제 `FedAvg.aggregate_fit` 입력으로 사용
전체 전송량에는 양방향 모델 프레임, 그래프 제어 정보의 두 홉 전달 및 압축기 가중치 배포를 포함

별도 프로세스의 수신기가 실제 TCP로 프레임을 받아 복원한 배열을 Flower 집계에 사용하는 검증도 제공

```sh
python MapReduce_Hierarchical_Federated_Analysis/scripts/verify_transformer_tcp.py --artifact results/codec/inference/codec_model.bin --output results/transformer_tcp_check
```

이 검증은 TCP 경로와 기본 경로의 모델 및 전송 장부를 대조
검산을 위해 반환하는 배열과 RPC 봉투 비용은 과학적 비교 대상인 모델 전달 비용과 별도로 기록

검증 손실로 체크포인트를 선택하며 기본 CLI는 시험 표본에 대한 예측을 수행하지 않음
설정을 확정한 확인 실험에서만 `--evaluate-test`를 추가
이미 사용한 결과 폴더를 덮어쓰지 않으며 새로운 실행은 별도 출력 폴더에 저장

`legacy`는 보고서의 기존 zlib/XOR 방식과 비교하기 위해 유지한 기준 조건이며 기본 제안 경로는 `transformer`
저장된 전송 장부, 제거 마스크와 모델 예측은 별도 검증기로 다시 확인

```sh
defai verify-run results/digits_transformer --output results/digits_transformer_check.json
python Graph_Dynamic_Deduplication/examples/compare_methods.py --artifact results/codec/inference/codec_model.bin --output results/comparison
```

반복 비교의 기본 구성은 두 데이터셋, 12개 시드, 네 가지 방식의 96회 실행
시험 평가를 수행하려면 고정된 설정을 확인한 후 비교 실행에도 `--evaluate-test`를 명시
`Graph_Dynamic_Deduplication/examples/plot_results.ipynb`는 생성된 `aggregate_metrics.csv`를 읽으며 색상과 글꼴을 별도로 지정하지 않음
노트북의 분석 및 그림 코드는 기존 Python 3.10 환경에서도 별도로 실행할 수 있도록 학습 패키지를 import하지 않음

짧은 실행 확인은 다음과 같이 수행
아래 설정은 파이프라인 진단용이며 80라운드 성능 실험을 대체하지 않음

```sh
defai collect-calibration --rounds 4 --drift-round 3 --max-blocks-per-role 256 --output results/quick_calibration
defai train-codec --epochs 2 --calibration results/quick_calibration --output results/quick_codec
defai benchmark --rounds 4 --drift-round 3 --artifact results/quick_codec/inference/codec_model.bin --output results/quick_run
```

## 평가 지표

```sh
defai metrics-schema
defai metrics --input predictions.npz --observations observations.json --output results/metrics.json
```

`predictions.npz`의 필드는 `labels`와 `probabilities`
추가 측정은 지표 정의의 `field` 이름을 키로 하는 JSON 객체로 전달
각 지표는 이름, 목적, 단위, 값 및 측정 상태를 반환하며 관측하지 않은 항목을 임의의 수치로 채우지 않음
분류 지표는 출력 클래스 전체의 혼동행렬에서 계산하므로 HAR의 6개 활동과 Digits의 10개 클래스를 모두 처리

목적별 실험은 다음과 같이 실행
생성 자료의 기본 위치는 현재 작업 폴더의 `results`이며 `DEFAI_WORKSPACE`로 작업 루트를 변경 가능

```sh
defai metric-study multimodal -- --download --prepare-only
defai metric-study multimodal
defai metric-study verify-multimodal
defai metric-study crossnode
defai metric-study cross-verify-multimodal
defai metric-study system -- --stage protocol
defai metric-study system -- --stage main
defai metric-study transport-echo
defai metric-study verify-system
defai metric-study export-system
defai metric-study recovery
defai metric-study privacy
defai metric-study privacy-control
defai metric-study verify-privacy
defai metric-study export-privacy
```

세부 실행 인수는 해당 모듈의 도움말로 확인
예: `python -m defai.metrics.studies.purpose_system_study --help`

## 공개 범위

이 폴더는 소스, 설정, 테스트와 문서만 공개하는 구성
원본 데이터, 원본 보고서 PDF, 모델 가중치, 분할 식별자, 예측 및 실험 결과는 배포하지 않음
실행 후 생성되는 `results`, `data`, `artifacts`와 모델 파일은 `.gitignore` 및 배포 스크립트에서 제외

```sh
python scripts/build_source_archive.py
```

배포 ZIP에는 파일별 SHA256 매니페스트를 포함하며 ZIP 검사와 원본 내용 대조를 수행
실험 결과를 새로 생성한 뒤에도 공개할 파일은 이 배포 스크립트로 선별

기본 연합 분석은 단일 호스트에서 논리 단말을 동기 실행하는 참조 구현
시스템 평가의 별도 프로세스 및 localhost TCP 검증과 물리적 다중 서버 또는 WAN 성능을 구분
유사도는 중복 식별 후보 선정에 사용하며 실제 제거는 동일성이 확인된 사본에 한정
프라이버시는 명시한 클리핑, 잡음, 인접성 및 공개 범위에 대한 계산과 공격 측정을 함께 확인
