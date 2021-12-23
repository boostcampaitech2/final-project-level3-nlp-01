# 다국어 채팅 서비스 앵무새 톡

<div align="center">
    <img src="https://i.imgur.com/145udIs.png" alt="Logo" width="300">
    <br>
    <h3 align="center">Run & Learn Team - BoostCamp AI 2nd</h3>
</div>


## 팀원 소개


| [강석민](https://github.com/Kangsukmin) | [김종현](https://github.com/gimmaru) | [김태현](https://github.com/taehyeonk) | [오동규](https://github.com/Oh-Donggyu) | [윤채원](https://github.com/ycw0363) | [허은진](https://github.com/eunaoeh) |
| :---:|:---:|:---:|:---:|:---:|:---:|
|<img src="https://avatars.githubusercontent.com/u/50981395?v=4" width=100 >| <img src="https://avatars.githubusercontent.com/u/87857169?v=4" width=100 > | <img src="https://avatars.githubusercontent.com/u/29690062?v=4" width=100> | <img src="https://avatars.githubusercontent.com/u/81454649?v=4" width=100 > |<img src="https://avatars.githubusercontent.com/u/80023607?v=4" width=100 > |<img src="https://avatars.githubusercontent.com/eunaoeh" width=100 > |

<hr><br>

<details>
  <summary>목차 보기</summary>
  <ol>
        <li>
            <a href="#프로젝트-개요">프로젝트 개요</a>
            <ul>
                <li><a href="#프로젝트-구성">프로젝트 구성</a></li>
                <li><a href="#데이터셋">데이터셋</a></li>
            </ul>
        </li>
        <li>
            <a href="#프로젝트-설명">프로젝트 설명</a>
            <ul>
                <li><a href="#다국어-번역-모델">다국어 번역 모델</a></li>
                <li><a href="#모델-경량화">모델 경량화</a></li>
            </ul>
        </li>
        <li>
            <a href="#Code">Code</a>
            <ul>
                <li><a href="#Code-Description">Code Description</a></li>
                <li><a href="#Usage">Usage</a></li>
                <li><a href="#서비스-실행방법">서비스 실행방법</a></li>
            </ul>
        </li>
        <li>
            <a href="#결과">결과</a>
        </li>
  </ol>
</details>

</br>




# 프로젝트 개요
## 프로젝트 구성
한국어, 영어, 중국어 번역이 가능한 다국어 번역 모델을 구현하고, 벡엔드와 프론트엔드를 각각 FastAPI와 Streamlit으록 구축하여 다국어 번역 채팅 서비스를 제공합니다.

<div align="center">
    <img src="https://i.imgur.com/vAD4sgm.png">
</div>



## 데이터셋
번역 데이터의 source language와 target language를 교차해서 양방향 번역 데이터로 사용했습니다.
```
학습데이터
- AI Hub 번역 데이터 
- 허깅페이스 Datasets 공개 데이터

평가데이터
- 허깅페이스 Datasets 공개 데이터
```

# 프로젝트 설명


## 다국어 번역 모델
**Modified Multi-way Graformer**
![](https://i.imgur.com/4pV9N8b.png)

Multy-way NMT의 특성과 [Multilingual Translation via Grafting Pre-trained Language Models(2021)](https://arxiv.org/pdf/2109.05256.pdf) 논문의 모델을 참고하여 Model Custom 했습니다. 인코더는 Multi-lingual BERT를 각 언어에 맞게 fine-tuning 하고 디코더는 Monolingual로 학습된 언어별 GPT 모델을 사용했습니다.

Encoder, decoder, grafting module은 아래와 같이 사용하였습니다.

- Encoder
    - Multi-lingual BERT (base)
- Decoder
    - Korean: [kykim/gpt3-kor-small_based_on_gpt2](https://huggingface.co/kykim/gpt3-kor-small_based_on_gpt2)
    - English: [gpt2](https://huggingface.co/gpt2)
    - Chinese: [ckiplab/gpt2-base-chinese](https://huggingface.co/ckiplab/gpt2-base-chinese)
- Grafting Module
    - num encoder layers: 2 (Bart encoder layer 적용)
    - num decoder layers: 2 (Bart decoder layer 적용)


## 모델 경량화
### DistilBERT + TinyBERT
- Embedding Layer Distillation
    - $L_{embd} = MSE(E^SW_e, E^T)$

- Transformer Layer Distillation
    - Teacher 모델의 각 hidden layer output 과 attention matrix를 이용했습니다.
    - $L_{hidn} = MSE(H_SW_h, H_T)$
    - $L_{attn} = \frac{1}{h}\Sigma_{i=1}^{h}{MSE(A_i^S, A_i^T)}$
- Prediction Layer Distillation
    - $L_{pred} = CE(z^T/t, z^S/t)$
    
### Weight Distillation
- Teacher 모델의 학습된 Weight를 Student 모델에 전이하고자 하는 시도를 하였습니다.
- Weight Transformation을 통해 Teacher 모델의 Weight를 Student 모델에 전달하였습니다.
- Weight Transformation에 활용되는 Matrix를 학습했습니다.


# 코드
## Code Description
### Configs
**Config Structure**
```
configs
├──data
|    └──default.yaml
├──decoder
|    ├──en_gpt.yaml
|    ├──ko_gpt.yaml
|    └──cn_gpt.yaml
├──encoder
|    └──bert.yaml
└──config.yaml
```
**Config Usage**

`config.yaml`의 `default.decoder`를 원하는 디코더 파일로 설정하여 사용할 수 있습니다. 그 외의 원하는 학습 parameter도 `config.yaml`에서 변경하여 사용합니다.

<!--`config.yaml`을 수정하여 모델 학습(train configuration 설정 및 data, encoder, decoder configuration의 entrypoint)-->

### Code
`train.py` : 모델을 훈련시키는 코드파일 입니다.

`utils.py` : 훈련에 필요한 유틸이 들어있는 코드파일 입니다.

`model.py` : 모델 구조를 결정하는 클래스 파일입니다.

`train_kd.py`: Knowledge Distillation을 하는 코드파일입니다.

`student_model.py`: Student 모델을 불러오는 클래스파일입니다.

`teacher_model.py`: Teacher 모델을 불러오는 클래스파일입니다.

`wd_student_config.py`: Weight Distillation을 통해 생성된 Student 모델의 Configuration을 지정하는 클래스파일입니다.

`wd_KdLoss_teacher_model.py`: TinyBERT의 Distillation loss를 적용할 수 있는 Teacher 모델을 불러오는 클래스파일입니다.

`wd_KdLoss_student_model.py`: TinyBERT의 Distillation loss를 적용할 수 있는 Student 모델을 불러오는 클래스파일입니다.

`wd_KdLoss_train.py`: TinyBERT의 Distillation loss를 적용한 Student 모델을 훈련시키는 코드파일입니다.

`wd_WdLoss_student_model.py`: Weight Distillation 논문의 loss를 적용할 수 있는 Student 모델을 불러오는 클래스파일입니다.

`wd_WdLoss_train.py`: Weight Distillation 논문의 loss를 적용한 Student 모델을 훈련시키는 코드파일입니다.

`loss.py`: TinyBERT의 Distillation loss를 구현한 클래스 파일입니다.

`evaluation.py`: 평가 데이터셋에 대하여 BLEU Score로 평가하는 코드 파일입니다.


## Usage
### Installation
```
$ poetry install && poetry update
```
### Training & Evaluation
1. Model Training
```
$ python train.py
```
2. Distillation
- Knowledg Distillation
```
$ python train_kd.py
```
- Weight Distillation
```
$ python wd_WdLoss_train.py

# tiny Bert에서 적용한 Loss function
# Out of Memory 문제
$ python wd_KdLoss_train.py
```

3. Evaluation
```
$ python evaluation.py
```

## 서비스 실행방법

**Poetry 설치 (venv)**
```shell=
// 설치
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

// 기본경로를 현재폴더로
poetry config virtualenvs.in-project true
poetry config virtualenvs.path "./.venv"

// 가상환경 설치
poetry install && poetry update

// 새로운 모듈 추가시
poetry add [모듈이름]
```

**Streamlit (FrontEnd)**

```shell=
cd client

poetry install && poetry update

streamlit run main.py --server.port 6006 -- --pwd [사용할 패스워드]
```

**FastAPI (Server)**

```shell=
cd server

poetry install && poetry update

python app/main.py
```

**FastAPI (Model Server)**

- 실행하시기 전에 학습된 모델을 각 언어폴더(en/zh/ko)에 넣어주세요.

```shell=
cd model_server

poetry install && poetry update

python main.py
```

### Weight Distillation 실행방법

**Student Model Layer 설정**
```
wd_student_config.py

StudentEncoderConfig 클래스의 __init__ 메서드 안에 있는 num_hidden_layers에 원하는 student 모델 레이어 개수 지정
StudentDecoderConfig 클래스의 __init__ 메서드 안에 있는 n_layer에 원하는 student 모델 레이어 개수 지정
```

# 결과
![](https://i.imgur.com/3WBZSkX.png)

- 자신이 사용하는 언어를 설정하면 자동적으로 해당언어로 번역해줍니다.
