# 다국어 채팅 서비스 앵무새 톡

<div align="center">
    <img src="https://i.imgur.com/145udIs.png" alt="Logo" width="300">

  <h3 align="center">Multilingual Messanger</h3>

  <p align="center">
    Run & Learn Team - BoostCamp AI Second
    <br />
  </p>
</div>

# 프로젝트 개요

![](https://i.imgur.com/vAD4sgm.png)

- 

# 기본 설정법

## Poetry 설치 (venv)

```
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
