FROM ghcr.io/astral-sh/uv:0.13.1 AS uv

FROM docker.io/python:3.14-slim-trixie AS build

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

COPY --from=uv /uv /bin/uv
COPY pyproject.toml uv.lock .python-version ./

RUN uv sync --locked --no-dev

FROM docker.io/python:3.14-slim-trixie

LABEL maintainer="Said Sef <saidsef@gmail.com> (saidsef.co.uk/)"
LABEL author="uk.co.saidsef.ml-classifier=v3.0"
LABEL org.opencontainers.image.source="https://github.com/saidsef/ml-classifier"
LABEL org.opencontainers.image.description="ML news classifier"

ENV PORT ${PORT:-7070}
ENV FLASK_APP "classifier-ml.py"
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

COPY --from=build /opt/venv /opt/venv
COPY classifier.py .
COPY classifier-ml.py .
COPY ./data/voting_classifier.pickle.xz data/voting_classifier.pickle.xz
COPY ./data/voting_classifier.pickle.xz.sha256sum data/voting_classifier.pickle.xz.sha256sum

RUN chown nobody -R /app

USER nobody

EXPOSE ${PORT}

HEALTHCHECK --interval=60s --timeout=10s CMD curl --fail http://localhost:${PORT}/ || exit 1

CMD ["classifier-ml.py"]
ENTRYPOINT ["python"]
