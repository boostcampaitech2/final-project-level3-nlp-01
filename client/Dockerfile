FROM python:3.8-slim-buster

ENV ROOT_PWD=password
EXPOSE 8501

WORKDIR /app

COPY poetry.lock pyproject.toml ./

RUN pip3 install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

COPY . . 

CMD [ "streamlit", "run", "main.py", "--", "--pwd", ROOT_PWD ]