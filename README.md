# Korean_Law
NLP를 이용한 한국어 법률 계약서 적격 여부 판별 프로젝트입니다. (2021.07 ~ 2021.10)

데이터는 부득이하게 공개할 수 없는 점 양해 부탁드립니다.

## Goal
한국어 법률 계약서에는 필수적으로 포함되어야 하는 조항들이 존재합니다.

특정 계약서에 대해 필수적으로 포함되어야 하는 조항들이 모두 존재하는지, 존재하지 않는다면 어떠한 조항이 누락되었는지 등을 기계적으로 확인할 수 있도록 모델을 학습합니다.

이 모델은 계약서의 적격 여부를 판별하여 공정한 계약 체결에 도움을 줄 수 있습니다.

## Method
**BERT를 이용한 Multi-label Classification 문제**로 전환하여 해결합니다.

BERT pretrained model로는 https://github.com/snunlp/KR-BERT-MEDIUM 의 **KR-BERT-MEDIUM**을 사용합니다.

각 조항에 대해 자주 사용되는 frequent keyword가 등장하는 것을 확인하였고, 따라서 keyword에 대한 각 조항에 대한 분포를 확률로 나타내어 만든 키워드 벡터를 concat하여 학습시킨 결과 성능이 향상되는 것을 확인하였습니다.

## Requirements
코드 실행에 사용한 GPU는 NVIDIA TITAN RTX 1대이고, CUDA version은 11.4 version입니다.

필요한 라이브러리 및 버전은 다음과 같습니다.
- torch == 1.9.1
- transformers == 4.10.3
- pytorch_lightning == 1.2.8

## Code Details
- `keyword2vec.py`
    - Frequent keyword들의 각 조항에 대한 분포를 계산하여 [키워드 - 임베딩 벡터] 대응 관계를 csv 파일로 저장
- `dataset.py`
    - input data에 따라 keyword vector 및 bert input id 등을 넘겨줄 수 있도록 처리하여 torch dataset 만듦
- `model.py`
    - input data에 따라 존재하는 키워드 벡터를 평균하여 만든 임베딩 벡터를 BERT를 통과하고 나온 마지막 hidden states의 CLS(또는 CLS+SEP)에 해당하는 위치의 벡터와 concat하고, 이를 다시 fully connected layer를 통과시키는 모델
- `main.py`
    - Train, evaluate
- `CLS_train.sh`, `CLS+SEP_train.sh`, `CLS_test.sh`, `CLS+SEP_test.sh`
    - 실행 파일

## Results
한국어 법률 계약서에 필수적으로 포함되어야 하는 조항들을 정답 라벨 list로 정하고 one-hot label 형태로 변환한 후, Multi-label Classification을 진행하였습니다.

BERT pretrained model을 이용하여 Fine-tuning 하는 방식의 실험과 학습 데이터로부터 키워드 벡터를 만들어 모델에 함께 이용하는 방식의 실험을 진행하였습니다.

그 결과, 키워드 벡터를 이용하여 학습시키는 방식이 성능이 더 좋게 나타나는 것을 확인하였습니다.

테스트 데이터를 통해 평가한 성능은 다음과 같이 나타납니다.
- Test Accuracy: 90.57 ~ 90.64
- Weighted Average Precision: 93.38 ~ 93.57
- Weighted Average Recall: 93.03 ~ 93.20
- Weighted Average F1-score: 93.26 ~ 93.27