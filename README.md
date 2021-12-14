# final-project-level3-nlp-01

# Poetry 설치 (venv)

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