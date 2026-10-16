# API reference

The service reads every request body as JSON, whatever its `Content-Type`, and answers in JSON.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | lists the routes and their methods |
| `GET` | `/api/v1/news` | returns a fixed health message |
| `POST` | `/api/v1/news` | classifies an article |
| `POST` | `/api/v1/train` | retrains the model on posted articles and replaces it |
| `GET` | `/metrics` | Prometheus metrics, see [operations](operations.md#metrics) |

## Classify an article

`POST /api/v1/news` takes the article text in `body`:

```json
{"body": "The central bank raised interest rates again as inflation stayed above target."}
```

```shell
curl -X POST http://localhost:7070/api/v1/news -H 'Content-Type: application/json' \
  -d @test/test.json
```

The response holds the predicted category and its probability under soft voting, a number between
0 and 1:

```json
{"category": "World", "score": 0.126}
```

The categories are the label strings from the dataset, see
[training the model](model.md#dataset). A payload without `body` returns a 500.

`GET /api/v1/news` returns `{"message": "healthy"}` without touching the model.

## Retrain the model

`POST /api/v1/train` takes two objects, `body` and `categories`, that map the same document IDs to
the article text and to its category:

```json
{
  "body": {"1": "First article text", "2": "Second article text"},
  "categories": {"1": "World", "2": "Opinion"}
}
```

The service fits a copy of the current pipeline on 80% of the posted documents and scores it on
the other 20%. Once the fit succeeds, the new model replaces the one in memory, and the response
reports the accuracy on the held-out documents with the size of each split:

```json
{"accuracy": 0.41, "test": 100, "train": 400}
```

| Status | Meaning |
| --- | --- |
| `200` | the new model is serving |
| `400` | the payload is malformed or too small to fit the pipeline, with the reason in `error` |
| `409` | another training run is in progress |

The new model learns from the posted documents alone and discards what the previous model
learned. The pipeline needs enough documents for its k-means step and its feature selection.
`test/train.json` shows the format, but its twelve documents are too few, and the request returns
a 400.

The request holds the connection open until the fit finishes. A fit over a few hundred documents
can outlast proxy timeouts such as the 60-second default in ingress-nginx. The fit carries on
after the client disconnects, and the new model replaces the old one when it completes.

The replacement stays in the memory of the process that received the request. A restart loads the
model from the image again, and other replicas keep serving their own copies. The endpoint has no
authentication, see [deploying on Kubernetes](deploying.md#exposure) for the routes each overlay
exposes.
