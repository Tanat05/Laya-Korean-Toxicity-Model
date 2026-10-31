# Laya Korean Toxicity Model

한국어 대화의 유해성을 9개 카테고리로 분류하도록 Convai Laya 다국어 ModernBERT를 파인튜닝하는 코드입니다. 원본 학습 노트북은 [Kaggle](https://www.kaggle.com/code/tanat05/laya-korean-toxicity-model-training), 데이터셋은 [Laya Korean Toxicity Fine-tuning Dataset](https://www.kaggle.com/datasets/tanat05/laya-korean-toxicity-finetune)에서 확인할 수 있습니다.

## 분류 카테고리

`sexual`, `sexual_minors`, `hate`, `harassment`, `threatening`, `violence`, `illicit`, `self_harm`, `spam` 및 통합 위반(`flagged`) 점수를 출력합니다.

## 설치

Python 3.10 이상과 학습용 GPU 사용을 권장합니다. PyTorch는 사용하는 CUDA 환경에 맞는 버전을 먼저 설치하고, 나머지 의존성은 다음과 같이 설치합니다.

```bash
python -m pip install -r requirements.txt
```

## 학습

Kaggle CLI를 설정한 뒤 JSONL 데이터셋을 내려받습니다.

```bash
kaggle datasets download -d tanat05/laya-korean-toxicity-finetune -p data --unzip
python train.py --data-dir data --output-dir laya-korean-toxicity-model
```

`train.py`는 하위 디렉터리를 포함해 `.jsonl` 파일을 읽습니다. 각 줄은 `text`와 `target.flagged`, `target.categories` 필드를 가진 레코드여야 합니다. 기본 설정은 70/15/15 학습·검증·테스트 분할, 최대 길이 128, 배치 크기 32, 3 epoch입니다.

학습이 끝나면 출력 디렉터리에 `model.pt`, `encoder/`, `tokenizer/`, `training_summary.json`이 생성됩니다. `model.pt`에는 분류 헤드와 인코더 가중치가 모두 저장됩니다.

## 사전 학습된 파인튜닝 모델 사용

가중치 파일은 약 1.23GB이므로 이 저장소에는 포함하지 않았습니다. Kaggle에 로그인하고 Kaggle CLI 인증을 설정한 후 노트북 실행 산출물을 내려받으세요.

```bash
kaggle kernels output tanat05/laya-korean-toxicity-model-training -p model
```

내려받은 `model/laya-korean-toxicity-model/` 디렉터리를 추론에 사용합니다.

```bash
python predict.py model/laya-korean-toxicity-model "이 문장을 분류해 주세요"
```

출력은 통합 위반 여부·점수와 카테고리별 확률을 담은 JSON입니다. Python 코드에서도 사용할 수 있습니다.

```python
from predict import predict

result = predict(
    "model/laya-korean-toxicity-model",
    "이 문장을 분류해 주세요",
)
print(result)
```

노트북 코드는 [Kaggle에서 직접 내려받기](https://www.kaggle.com/code/tanat05/laya-korean-toxicity-model-training) 할 수 있습니다. 해당 노트북은 GPU T4 x2 환경에서 학습했으며, 학습 데이터 31,191개를 train 70%, validation 15%, held-out test 15%로 나눕니다.

## 결과 및 주의사항

노트북의 held-out test 결과에서 9개 카테고리 macro F1은 약 0.09였으며, `harassment` 외 다수의 희소 카테고리는 재현율이 낮았습니다. 이 모델은 실험용이며, 실제 서비스에 적용하기 전에는 사용 목적에 맞는 독립 데이터로 평가하고 임계값을 조정해야 합니다. 출력 점수만으로 자동 제재를 결정하지 마세요.

기반 모델은 [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual)이며, 원본 노트북에서 사용하는 [Convai Laya](https://github.com/convaiinnovations/laya)의 라이선스와 데이터셋 이용 조건도 확인하세요.
