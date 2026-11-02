# Graph Dynamic Deduplication

로컬 단말 구조의 연관성을 반영한 그래프 기반 동적 중복 분석 제거 모듈

로컬 단말과 군집 중심의 연결 관계를 GraphSAGE로 학습하고, 데이터 변화에 따라 중복 식별 후보 그래프를 갱신
후보 단말 간 해시와 배열의 동일성을 검증하여 중복 사본의 반복 처리를 줄이고 식별 정보와 제거 결과를 재사용

## 주요 기능

- 데이터 요약과 단말별 데이터량을 반영한 가중 그래프 구성
- 가중 이웃 샘플링과 풀링 집계 및 정표본과 부정 표본 손실 최적화
- 학습된 특징 벡터와 공유 공간 유사도를 결합한 후보 이웃 선정
- 단말 요약과 데이터량 변화에 따른 그래프 및 특징 벡터 갱신
- SHA256과 특징 및 레이블 배열의 일치 검증을 통한 정확 중복 제거
- 데이터 버전과 후보 연결 관계를 확인하는 해시 및 제거 마스크 캐시
- 제거 이후 단말별 학습 스텝 변화에 대한 학습률 보정과의 연계

## 소스 위치

- [graph.py](src/defai/federated/graph.py): 실제 GraphSAGE 연관성 학습과 후보 그래프 구성 및 정확 일치 검증
- [cache.py](src/defai/federated/cache.py): 동적 갱신과 버전별 식별 정보 및 제거 결과 재사용
- [snapshots.py](src/defai/federated/snapshots.py): 불변 학습 스냅샷과 digest 및 ownership 제어 메시지
- [data.py](src/defai/federated/data.py): 고정 학습, 검증 및 시험 분할과 단말 데이터 변화 조건
- [learning.py](src/defai/federated/learning.py): 공유 Flower 클라이언트와 학습 및 고유 표본 보존 검사
- [examples/compare_methods.py](examples/compare_methods.py): 동일 조건의 기준 전송, compact 전송, 정적 및 동적 Transformer 방식 비교
- [examples/plot_results.ipynb](examples/plot_results.ipynb): 비교 실행 CSV를 읽는 기본 Matplotlib 시각화

주요 API는 `GraphDeduplicator`, `GraphSAGECachedDeduplicator`와 `ImmutableTrainingSnapshot`
`ImmutableTrainingSnapshot.from_arrays`로 구성한 스냅샷을 `GraphSAGECachedDeduplicator.apply`에 라운드별로 전달
입력이 바뀌지 않은 경우 캐시를 재사용하고, 요약과 데이터량이 달라지면 동적 모드에서 단말 관계를 다시 학습

공유 데이터와 학습 도구는 다계층 분석 및 평가 지표에서도 같은 구현을 사용
다계층 집계와 Transformer 전송은 [MapReduce 모듈](../MapReduce_Hierarchical_Federated_Analysis/README.md)과 연결
유사도는 후보 선정에 사용하며 실제 삭제는 동일성이 확인된 사본에 한정

## 설치와 비교 실행

이 모듈 폴더 안에서 시작하는 경우 상위 `Federated_Learning`으로 이동하여 세 모듈을 한 번 설치
실행 확인 환경은 Python 3.12

```sh
cd ..
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[transformer,plots]"
defai benchmark --dataset digits --artifact results/codec/inference/codec_model.bin --output results/graph_dynamic
defai benchmark --dataset digits --artifact results/codec/inference/codec_model.bin --static --output results/graph_static
python Graph_Dynamic_Deduplication/examples/compare_methods.py --artifact results/codec/inference/codec_model.bin --output results/comparison
```

압축기 파일은 [MapReduce 모듈의 학습 절차](../MapReduce_Hierarchical_Federated_Analysis/README.md#설치와-실행)로 먼저 생성
두 실행은 같은 전송 설정을 사용하고 그래프의 동적 갱신 여부를 비교
전송량과 함께 변경 이후 중복 사본 제거율, 학습량과 분류 성능을 확인

API와 통합 실행 안내는 [상위 README](../README.md), 기술 범위는 [모듈 범위](../docs/module_scope.md), 라이선스는 [MIT License](../LICENSE) 참조
