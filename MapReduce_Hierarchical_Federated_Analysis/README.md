# MapReduce Hierarchical Federated Analysis

MapReduce 기술을 활용한 효율적인 다계층 연합 분석 모듈

클라이언트, 그룹, 중앙 서버의 계층 구조에서 분석 결과를 블록과 키값쌍으로 구성하고 표본수에 따라 집계
Transformer로 압축한 잠재 표현과 주요 잔차를 전송하고, 수신 측에서 복원한 업데이트를 실제 Flower 집계에 반영

## 주요 기능

- 그룹과 블록 식별자를 키로 사용하는 표본수 가중 MapReduce
- 개수, 합과 제곱합에 대한 충분통계 집계
- 실제 Transformer 인코더와 디코더를 이용한 업데이트 압축 및 복원
- INT8 잠재 표현, 상위 8개 INT8 잔차와 블록 이득 보정
- 압축기 학습, 수신기 배포 및 양방향 전송량 기록
- 직접 가중 평균과 독립 shuffle 집계 및 저장 결과 검산

## 소스 위치

- [aggregation.py](src/defai/aggregation.py): 블록 MapReduce와 충분통계 집계
- [compression/model.py](src/defai/compression/model.py): Transformer 모델과 학습 및 추론용 내보내기
- [compression/transformer.py](src/defai/compression/transformer.py): 잠재 표현과 잔차의 부호화 및 수신 복원
- [compression/runner.py](src/defai/compression/runner.py): 실제 수신 배열을 사용하는 그룹 및 중앙 Flower 집계
- [compression/calibration.py](src/defai/compression/calibration.py): 학습 표본만 사용하는 압축기 학습용 업데이트 수집
- [audit.py](src/defai/audit.py): 저장된 전송 장부, 제거 마스크와 모델 예측의 독립 검산
- [scripts/verify_transformer_tcp.py](scripts/verify_transformer_tcp.py): 별도 프로세스가 실제 TCP로 수신한 업데이트의 복원 및 Flower 반영 검증

단말 관계와 중복 제거는 [Graph Dynamic Deduplication](../Graph_Dynamic_Deduplication/README.md)의 구현을 재사용
공통 클라이언트와 데이터 처리 코드를 복사하지 않고 같은 Python 패키지에서 공유
기본 무손실 대조군은 compact zlib/XOR이며 legacy zlib/XOR는 기존 보고서와의 비교에 필요한 기준 방식으로 유지

## 설치와 실행

세 모듈은 상위 `Federated_Learning` 폴더에서 한 번 설치
이 모듈 폴더 안에서 시작하는 경우 아래와 같이 상위 폴더로 이동
실행 확인 환경은 Python 3.12

```sh
cd ..
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[transformer,plots]"
defai verify-aggregation --full --output results/aggregation_check.json
defai collect-calibration --output results/calibration
defai train-codec --calibration results/calibration --output results/codec
defai benchmark --dataset digits --artifact results/codec/inference/codec_model.bin --output results/digits_transformer
defai verify-run results/digits_transformer --output results/digits_transformer_check.json
python MapReduce_Hierarchical_Federated_Analysis/scripts/verify_transformer_tcp.py --artifact results/codec/inference/codec_model.bin --output results/transformer_tcp_check
```

기본 벤치마크는 단말 12개, 그룹 4개, 80라운드이며 41라운드에 데이터 구성 변경
검증 손실로 모델을 선택하고, 시험 예측은 설정 확정 이후 `--evaluate-test`를 명시한 실행에서만 수행
실행 결과와 학습 가중치는 소스 공개 대상에서 제외

전체 설치와 비교 실행은 [상위 안내](../README.md), 기술 범위는 [모듈 범위](../docs/module_scope.md), 라이선스는 [MIT License](../LICENSE) 참조
