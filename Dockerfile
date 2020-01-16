FROM python:3-slim

LABEL maintainer="saidsef@gmail.com"

ENV PORT ${PORT:-7070}
ENV version 1.0

WORKDIR /app

COPY classifier.py .
COPY classifier-ml.py .
COPY requirements.txt .
COPY ./data/rfc.pickle data/rfc.pickle

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s CMD curl --fail http://localhost:${PORT}/ || exit 1

CMD ["classifier-ml.py"]
ENTRYPOINT ["python"]
