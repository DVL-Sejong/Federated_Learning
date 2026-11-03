# 중간층 매칭 기반 지식 증류 모듈 (`layer_kd`)

상위 계층의 교사 모델(전역 모델)이 가진 지식을 하위 계층의 소규모 학생 모델에 전수합니다. 교사·학생의 중간층을 자동으로 짝지어(해상도별 스테이지 + 분류기 직전 특징) 로짓·특징맵·어텐션·GAN 매칭 손실을 조합합니다.
중앙 증류와, 클라이언트가 로컬 데이터로 학생을 증류하고 서버가 학생을 집계하는 연합 증류(`TeacherStrategy` + KD ClientApp 핸들러)를 제공합니다.

## 연차별 개발 항목 통합

| 연차 | 개발 항목 | 방법(`--methods`) | 처리 |
|---|---|---|---|
| 1차 | 지식 증류 설계: 교사 로짓 소프트 라벨 + 실제 라벨 혼합 손실 | `logit` | **유지** (모든 증류 방법의 기본 항) |
| 1차 | 다수 교사 예측 확률 가중 결합 앙상블 | `ensemble` | **유지** (앙상블 교사, `teacher_weights`) |
| 2차 | 중간 레이어 특징맵 매칭(L2), 레이어 중요도 가중 다중 매칭(Weighted FM), 중요도에 따른 적응형 학습률 | `wfm` | **유지**: 깊이에 비례한 가중치 `w_l`, 1×1 Conv/Linear 어댑터, 스테이지별 학습률 `lr·max(0.5, n·w_l)` |
| 3차 | 어텐션 맵 매칭(손실: 어텐션 차이, 이득: 상호 엔트로피 기반 정보량) | `at` | **최신 구현** |
| 3차 | GAN 기반 중간층 매칭(교사 중간층 출력을 실제 라벨, 노이즈 입력 생성자, 판별자) | `gan` | **최신 구현** |
| 3차 | 고도화된 중간층 매칭 지식 증류 | `combined` | 로짓 + 가중 특징맵 + 어텐션(손실+이득) + GAN 매칭 |

`student`는 증류 없이 학습한 비교 기준입니다.

## 손실

* 로짓: `(1−α)·CE + α·T²·KL(p_t^T ‖ p_s^T)` (앙상블은 교사 확률의 가중 평균).
* 가중 특징맵: `Σ_l w_l · MSE(LN(adapter_l(F_s^l)), LN(F_t^l))`.
* 어텐션: 손실 `(β/2)·Σ‖a_s − a_t‖²` (a = 채널 제곱 평균 공간맵의 L2 정규화) + 이득 `λ·KL(P_t ‖ P_s)` (P = 표준화 어텐션의 softmax, 상호 엔트로피 최소화 = 공유 정보량 최대화).
* GAN: 생성자 `G(F_s, z)`(1×1 Conv/Linear, 노이즈 채널), 스펙트럴 정규화 판별자, LSGAN 손실 + 재구성 앵커. 판별자(Adam)와 학생·생성자(SGD)를 매 스텝 교대로 갱신.
* 학습 안정화를 위해 그래디언트 노름 클리핑(`clip`)을 적용합니다. 어댑터·생성자·판별자는 학습 전용이며 배포 학생 모델에 포함되지 않습니다.

## 실행

```bash
python -m layer_kd central --teachers 2 --student-width 0.25 --epochs 3
python -m layer_kd central --methods student,logit,wfm,at,gan,combined
python -m layer_kd federated --clients 6 --rounds 5 --methods student,combined
python -m layer_kd federated --backend kist --methods student,combined
```

Python API:

```python
from layer_kd.trainer import KDConfig, KDTrainer
trainer = KDTrainer(KDConfig(method="combined"), [teacher], student, input_shape=(1, 1, 28, 28))
logs, steps = trainer.fit(x, y, epochs=2, lr=0.02)
```

연합 증류: `make_kd_handler(backend, KDConfig(...), student_width)`를 ClientApp 학습 핸들러로, `TeacherStrategy(strategy, teacher_arrays)`로 교사 가중치를 학습 메시지에 첨부합니다. KIST 백엔드에서는 KIST의 데이터 분할·모델 계열(KIST `Net`, 학생은 채널 폭 축소)을 사용하고 집계는 KIST 전략이 수행합니다.

## 출력

`results/layer_kd/kd_central.json`(교사·학생별 파라미터 수, 지연시간, 정확도, 크기 비율, 손실 항), `kd_federated.json`(방법별 라운드 정확도).

## 측정 결과

측정 환경: 이 저장소의 `run_demo.py` 기본 설정(synthetic 데이터, 3라운드 내외의 짧은 실행, CPU i9-11900F). 짧은 실행이므로 절대값보다 같은 설정 안의 비교로 해석합니다.

중앙: `python -m layer_kd central --teacher-epochs 2 --epochs 2` (교사 2개 정확도 0.939 / 0.972, 1,663,370 파라미터; 학생 폭 0.25, 105,194 파라미터 = 교사의 6.3%, 학습 데이터 20%)

| 방법 | student | logit | ensemble | wfm | at | gan | combined |
|---|---|---|---|---|---|---|---|
| 테스트 정확도 | 0.208 | 0.251 | 0.267 | 0.420 | 0.609 | 0.229 | 0.512 |

3 에폭 실행에서는 student 0.341 / logit 0.607 / ensemble 0.636 / wfm 0.684 / at 0.730 / combined 0.751이었습니다. 짧은 학습에서 GAN 단독 매칭은 판별자 학습이 안정되기 전이라 효과가 작습니다.

연합: `python -m layer_kd federated --methods student,logit,combined --rounds 3 --clients 6` (교사 0.939)

| 방법 | 라운드별 정확도 | 최종 |
|---|---|---|
| student | 0.088, 0.147, 0.702, 0.841 | 0.841 |
| logit | 0.088, 0.186, 0.715, 0.804 | 0.804 |
| combined | 0.088, 0.665, 0.802, 0.879 | 0.879 |
