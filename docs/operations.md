# Operations

## Health checks

| Check | Request | Defined in |
| --- | --- | --- |
| Docker health check | `GET /` every 60 seconds | `Dockerfile` |
| Kubernetes readiness | `GET /metrics` on port 7070 | `deployment/base/deployment.yml` |
| Kubernetes liveness | a TCP connection to port 7070 | `deployment/base/deployment.yml` |

`GET /api/v1/news` returns `{"message": "healthy"}` and suits an external check. None of these
requests touch the model. A failed model load stops the process at start, so a running process
always has a model.

## Metrics

`GET /metrics` serves Prometheus metrics from
[prometheus-flask-exporter](https://github.com/rycus86/prometheus_flask_exporter). Request
metrics carry the route rule in the `url_rule` label, rather than the raw path.

| Metric | Type | Labels |
| --- | --- | --- |
| `flask_http_request_duration_seconds` | histogram | `method`, `status`, `url_rule` |
| `flask_http_request_total` | counter | `method`, `status` |
| `flask_http_request_exceptions_total` | counter | `method`, `status` |
| `flask_exporter_info` | gauge | `version` |

The Prometheus client adds the standard `process_*` and `python_*` metrics. A training run shows
as a `POST` to `/api/v1/train` in the duration histogram.

## Common failures

| Symptom | Cause |
| --- | --- |
| `FileNotFoundError: .../data/voting_classifier.pickle.xz is missing` at start | the model file is absent, see [running the service](running.md#source-checkout) |
| an error or `InconsistentVersionWarning` while unpickling at start | the model was written by another scikit-learn version than the one installed |
| `400` from `/api/v1/train` with `n_samples=... should be >= n_clusters=10` | too few documents for the k-means step |
| `400` from `/api/v1/train` with `KeyError` | a document ID in `body` has no entry in `categories`, or a key is missing |
| `409` from `/api/v1/train` | another training run has not finished |
| a `500` from `POST /api/v1/news` | the payload has no `body` key |
| the pod restarts with `OOMKilled` during training | the memory limit has no room for a second copy of the model |
| predictions differ between replicas | one replica was retrained, see [retrain the model](api.md#retrain-the-model) |
