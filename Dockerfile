FROM docker.io/python:3.11-slim-buster

LABEL maintainer="Said Sef <saidsef@gmail.com> (saidsef.co.uk/)"
LABEL author="uk.co.saidsef.ml-classifier=v3.0"
LABEL org.opencontainers.image.source="https://github.com/saidsef/ml-classifier"
LABEL org.opencontainers.image.description="ML news classifier"

ENV PORT ${PORT:-7070}
ENV FLASK_APP "classifier-ml.py"

WORKDIR /app

COPY classifier.py .
COPY classifier-ml.py .
COPY requirements.txt .
COPY ./data/randomforestclassifier.pickle.xz data/voting_classifier.pickle.xz
COPY ./data/randomforestclassifier.pickle.xz.sha256sum data/voting_classifier.pickle.xz.sha256sum

RUN pip install --no-cache-dir -r requirements.txt && \
    chown nobody -R /app

USER nobody

EXPOSE ${PORT}

HEALTHCHECK --interval=60s --timeout=10s CMD curl --fail http://localhost:${PORT}/ || exit 1

CMD ["classifier-ml.py"]
ENTRYPOINT ["python"]
