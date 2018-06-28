FROM python:2-slim
MAINTAINER Said Sef <saidsef@gmail.com> (saidsef.co.uk/)

ARG PORT=""

ENV PORT ${PORT:-7070}
ENV version 1.0

WORKDIR /app

COPY news-ml.py .
COPY requirements.txt .
COPY ./data/lsvc.pickle data/lsvc.pickle

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE ${PORT}

CMD ["python", "news-ml.py"]
