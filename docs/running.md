# Running the service

The service runs from the published image or from a source checkout. Both start the Werkzeug
development server that `app.run()` provides, listening on every interface.

## Container

```shell
docker run -d -p 7070:7070 docker.io/saidsef/ml-classifier:latest
```

The image listens on port 7070 and runs as `nobody`. Docker checks its health every 60 seconds
with a request to `GET /`. [CI and releases](ci.md#image-tags) lists the published tags.

## Source checkout

Git does not track the model, so a fresh checkout has no `data/voting_classifier.pickle.xz`. The
service stops at start with a `FileNotFoundError` that names the missing file. Copy the model out
of an image built from the same commit:

```shell
docker create --name ml-classifier docker.io/saidsef/ml-classifier:latest
docker cp ml-classifier:/app/data/voting_classifier.pickle.xz data/
docker rm ml-classifier
```

[Running the notebook](model.md#running-the-notebook) trains a model instead and writes it into
`data/`.

The image and the checkout install scikit-learn from the same `uv.lock`. scikit-learn does not
guarantee that a pickle loads under another version, so a model copied from an image built from a
different commit may fail to load.

With the model in place:

```shell
uv sync
uv run classifier-ml.py
```

## Configuration

| Variable | Default | Effect |
| --- | --- | --- |
| `PORT` | `7070` | the port the server listens on |

The model path is fixed at `data/voting_classifier.pickle.xz`, relative to `classifier.py`.
