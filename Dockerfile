FROM python:3.7

LABEL version="2.0"
LABEL maintainer="Said Sef said@saidsef.co.uk (saidsef.co.uk/)"

ARG PORT=""

ENV PORT ${PORT:-7070}
ENV version 2.0

WORKDIR /app

COPY classifier.py .
COPY classifier-ml.py .
COPY requirements.txt .
COPY ./data/lsvc.pickle data/lsvc.pickle

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE ${PORT}

CMD ["python3", "classifier-ml.py"]
