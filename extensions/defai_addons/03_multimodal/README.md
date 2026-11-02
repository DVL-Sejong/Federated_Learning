# Transformer 기반 멀티모달 학습 모델

1차년도 공동 임베딩과 피처 융합, 2차년도 어텐션 기반 연결을 3차년도 Transformer 구조에 통합했습니다. 기본 모델은 `MultimodalTransformer`이며 이전 완성 모델은 `defai_addons/multimodal/baselines`에 보존합니다.

이미지 CNN, 텍스트 Transformer, 오디오 CNN 및 GRU의 출력을 공동 공간에 사상합니다. 모달리티 가중치와 위치 정보를 통합한 뒤 Transformer 인코더, 크로스모달 디코더, GMU와 계층적 피처 융합으로 분류 확률 및 표현 벡터를 생성합니다. 생성 디코더는 `sequence_vocab`을 지정할 때만 활성화합니다.

## 설치 및 기본 실험

상위 폴더에서 `python -m pip install ".[learning]"`를 실행합니다.

이 작업 공간에서 다음과 같이 실행할 수 있습니다.

```powershell
python run.py --part central --epochs 5 --n-train 600 --n-test 120
```

기본 실험 데이터는 이미지, 토큰 및 파형으로 구성된 합성 데이터입니다. 기본 실행 결과는 실제 서비스 데이터의 성능을 보장하지 않습니다. 연합 학습 보조 실험은 `python run.py --part federated --help`로 확인합니다.

## 외부 Python 코드에서 모델 호출

```python
from defai_addons.api import create_multimodal_model, predict_multimodal

model_config = {"d": 64, "n_classes": 6, "enc_layers": 2, "dec_layers": 1}
model = create_multimodal_model(**model_config)
result = predict_multimodal(model, batch)
print(result["probabilities"].shape)   # [N, 6]
print(result["representation"].shape)  # [N, 192]
```

위의 새 모델은 미학습 상태입니다. 실제 추론에는 아래 방법으로 학습한 체크포인트를 불러옵니다. `batch`는 호출자가 준비한 다음 전처리 배열 또는 텐서입니다.

| 키 | 형태 및 의미 |
|---|---|
| `image` | float `[N,1,28,28]`, 학습 때와 같은 정규화 |
| `text` | 정수 `[N,16]`, 토큰 ID 0~63, 0은 padding |
| `audio` | float `[N,1,800]`, 학습 때와 같은 정규화 |
| `mask` | bool `[N,3]`, 이미지, 텍스트, 오디오 순서, 행마다 하나 이상 True |
| `y` | 학습 및 평가 때 필요한 정수 `[N]` 클래스 번호 |

문자열 토큰화와 실서비스 오디오 전처리는 자동 수행하지 않습니다. 현재 텍스트 인코더는 실험용 64개 어휘를 사용하므로 임의의 자연어를 그대로 넣을 수 없습니다. 결측 모달리티도 해당 배열 형태를 유지하고 `mask=False`로 표시합니다.

## 학습한 모델 저장 및 재호출

```python
from defai_addons.api import (create_multimodal_model, save_multimodal_checkpoint,
                              load_multimodal_checkpoint, predict_multimodal)
from defai_addons.multimodal.train import fit, evaluate

config = {"d": 64, "n_classes": 6}
model = create_multimodal_model(**config)
training = fit(model, train_data, validation_data, epochs=10, batch_size=32)
metrics = evaluate(model, test_data, "cpu")
save_multimodal_checkpoint(model, "C:/experiments/multimodal.pt", config)
loaded = load_multimodal_checkpoint("C:/experiments/multimodal.pt")
prediction = predict_multimodal(loaded, test_data)
```

`train_data`, `validation_data`, `test_data`는 위 형식의 배열 사전이며 서로 분리된 표본을 사용합니다. `fit`은 마지막 epoch의 모델 상태를 남깁니다. 체크포인트에는 모델 설정과 가중치가 들어갑니다. 저장 시 기존 파일을 덮어쓰지 않습니다.

## 별도 프로세스에서 추론

```python
import numpy as np
np.savez("C:/experiments/multimodal-input.npz",
         **{k: test_data[k] for k in ("image", "text", "audio", "mask")})
```

```powershell
defai-addons multimodal-predict --checkpoint C:/experiments/multimodal.pt --data C:/experiments/multimodal-input.npz --out C:/experiments/predictions.json
```

JSON에는 `logits`, `probabilities`, `representation`이 포함됩니다. 학습 과정에 대한 실험 기록은 `fit`의 반환값과 `evaluate` 결과로 별도 저장합니다.

## 기존 GitHub 시스템에 연결하는 경계

외부 Python 연결 스크립트에서 모델 생성, 학습 및 추론을 호출하거나 위 CLI를 별도 프로세스로 실행할 수 있습니다. 기존 저장소의 `defai` 패키지, 이미지 모델 및 CLI를 덮어쓰지 않습니다.

기존 Flower ServerApp은 이미지 모델 및 입력 형식을 직접 사용하므로 이 모델을 설정 문자열만으로 교체할 수 없습니다. 시스템 원본을 수정하지 않는 조건에서는 별도 멀티모달 워크플로로 호출합니다. 운영 Flower 클라이언트에 자동 연결하거나 기존 CNN 체크포인트를 변환하는 기능은 포함하지 않습니다.

검증 기록은 상위 폴더의 `integration_tests.log`에 있습니다. 모델 호출, 저장 후 복원한 출력의 일치, 출력 차원 및 잘못된 결측 마스크 거부를 확인합니다. 이 검사는 미학습 가중치를 사용하는 동작 검사이며 학습 정확도 검증과 구분합니다. 이전 학습 결과는 `evidence`에 보존했습니다.
