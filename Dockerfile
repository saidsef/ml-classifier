FROM python:3-slim

LABEL maintainer="Said Sef <saidsef@gmail.com> (saidsef.co.uk/)"
LABEL author="uk.co.saidsef.ml-classifier=v3.0"

ENV PORT ${PORT:-7070}
ENV VERSION 4.0
ENV MODEL v2021.08

WORKDIR /app

COPY classifier.py .
COPY classifier-ml.py .
COPY requirements.txt .
ADD https://github.com/saidsef/ml-classifier/releases/download/${MODEL}/randomforestclassifier.pickle.xz data/randomforestclassifier.pickle.xz

RUN pip install --no-cache-dir -r requirements.txt && \
    chown nobody -R /app

USER nobody

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s CMD curl --fail http://localhost:${PORT}/ || exit 1

CMD ["classifier-ml.py"]
ENTRYPOINT ["python"]
