# Deploying on Kubernetes

The manifests under `deployment/` are plain Kustomize. They set no namespace, so they apply to the
namespace of the current context unless `-n` names another.

## Layout

```text
deployment/
├── kustomization.yml   # default entry point: base and the image tag
├── base/               # Deployment and Service
├── ingress-nginx/      # base and an nginx Ingress
└── gateway-api/        # base and a Gateway API HTTPRoute
```

| Entry point | Resources | Needs |
| --- | --- | --- |
| `deployment/` | Deployment and Service | nothing beyond the cluster |
| `deployment/ingress-nginx/` | Deployment, Service and Ingress | an ingress controller for the `nginx` class |
| `deployment/gateway-api/` | Deployment, Service and HTTPRoute | the Gateway API CRDs and a Gateway to attach to |

```shell
kubectl create namespace classifier
kubectl apply -n classifier -k deployment/
kubectl rollout status deployment/classifier -n classifier
```

## Image tag

`deployment/kustomization.yml` sets the image tag for the default entry point through its
`images` override. Both overlays build on `base/` directly, so they take the tag written in
[`base/deployment.yml`](https://github.com/saidsef/ml-classifier/blob/main/deployment/base/deployment.yml).
A tag change belongs in both files. [CI and releases](ci.md#image-tags) lists the tags.

## Overlay settings

Each overlay carries values for a particular cluster. Replace them before applying it.

| Overlay | Setting | File |
| --- | --- | --- |
| ingress-nginx | the host in the Ingress rule | [`ingress-nginx/ingress.yml`](https://github.com/saidsef/ml-classifier/blob/main/deployment/ingress-nginx/ingress.yml) |
| gateway-api | the Gateway, its namespace and its listener in `parentRefs` | [`gateway-api/route.yml`](https://github.com/saidsef/ml-classifier/blob/main/deployment/gateway-api/route.yml) |

## Exposure

The overlays expose different parts of the API:

| Overlay | Paths routed |
| --- | --- |
| ingress-nginx | every path, with CORS enabled, including `/api/v1/train` and `/metrics` |
| gateway-api | `/api/v1/news` only |

`POST /api/v1/train` has no authentication and replaces the model a replica serves, see
[retrain the model](api.md#retrain-the-model). Behind the ingress-nginx overlay, any client that
reaches the host can call it.

## Pod

The Deployment runs one replica of the `classifier` container on port 7070, named `http`.

| Setting | Value |
| --- | --- |
| Readiness probe | `GET /metrics` on port 7070 |
| Liveness probe | a TCP connection to port 7070 |
| User | UID and GID 65534, with privilege escalation off |
| Filesystem | read-only root, with an `emptyDir` mounted at `/tmp` |
| Capabilities | all dropped, with the `RuntimeDefault` seccomp profile |
| Service account token | not mounted |
| Prometheus | `prometheus.io/scrape` and `prometheus.io/port` annotations on the pod template |

`base/deployment.yml` sets the CPU and memory requests and limits. Each replica holds the model in
memory, and a training run holds a second copy until it finishes, so the memory limit needs room
for both.
