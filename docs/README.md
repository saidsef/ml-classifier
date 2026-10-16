# ml-classifier

ml-classifier assigns a news article to a category. A scikit-learn pipeline, trained in a Jupyter
notebook on a dataset of labelled news articles, sits behind a Flask API that takes the text of an
article and returns the most probable category with its probability.

The trained model ships inside the container image. The Kustomize manifests run that image on
Kubernetes as a Deployment and a Service, with overlays for ingress-nginx and the Gateway API.

## Components

| Component | Detail |
| --- | --- |
| Notebook | `classifier-ml.ipynb` prepares the data, trains the pipeline and writes the model |
| Model | `data/voting_classifier.pickle.xz`, a soft-voting ensemble over TF-IDF features |
| Service | `classifier-ml.py`, a Flask application on port 7070 with Prometheus metrics |
| Image | `docker.io/saidsef/ml-classifier`, built by CI with the model inside |
| Manifests | Kustomize under `deployment/`, a base and two routing overlays |

## Requirements

| Requirement | Value |
| --- | --- |
| Container runtime | Docker or another OCI runtime, for the image and for training |
| Python | 3.13 or later, with [uv](https://docs.astral.sh/uv/), for a source checkout |
| Git LFS | for the dataset and the test payloads |
| Kubernetes | `kubectl` with Kustomize support, for a cluster deployment |

## Quick start

```shell
docker run -d -p 7070:7070 docker.io/saidsef/ml-classifier:latest

curl -X POST http://localhost:7070/api/v1/news -H 'Content-Type: application/json' \
  -d '{"body": "The central bank raised interest rates again as inflation stayed above target."}'
```

The response names the category and the probability the model gives it:

```json
{"category": "World", "score": 0.1122}
```

A source checkout needs the model file before the service starts, see
[running the service](running.md#source-checkout).

## Documentation

| Page | Contents |
| --- | --- |
| [Architecture](architecture.md) | How the notebook, the image and the cluster fit together, with a diagram |
| [Training the model](model.md) | The dataset, the pipeline, and how to run the notebook |
| [Running the service](running.md) | The container, a source checkout, and configuration |
| [API reference](api.md) | Every endpoint, with payloads and responses |
| [Deploying on Kubernetes](deploying.md) | The manifests, the overlays, and what to change in each |
| [Operations](operations.md) | Health checks, metrics, and what the common failures mean |
| [CI and releases](ci.md) | What CI builds and tests, the image tags, and how releases are cut |

## Repository

Source code and releases: [github.com/saidsef/ml-classifier](https://github.com/saidsef/ml-classifier)
