# CI and releases

Three GitHub Actions workflows build, tag and release the project. Each one starts when the one
before it succeeds on `main`.

## CI

`.github/workflows/ci.yml` runs on pushes and pull requests to `main` that touch the application,
the `Dockerfile`, `pyproject.toml`, `uv.lock`, `.python-version`, `test/`, JSON files in `data/`
or the workflow itself.
Changes to the documentation alone do not start it.

| Job | Runs for | Does |
| --- | --- | --- |
| `analyze` | every run | CodeQL analysis of the Python code |
| `review` | pull requests | dependency review through a shared workflow |
| `build` | every run | trains the model, builds and pushes the image, and scans it with Trivy |
| `k8s-test` | pull requests | applies `deployment/` to a kind cluster, in the `web` namespace |
| `auto-approve` | pull requests | approves the pull request once `build` and `k8s-test` pass |

The `build` job trains the model with `scripts/run.sh` inside the Jupyter data science image, as
described in [running the notebook](model.md#running-the-notebook), and writes a SHA-256 checksum
for each file in `data/` that ends in `.xz`. The image build copies the new model in. Trivy scans
the repository configuration and the pushed image, and the image results go to GitHub code
scanning. Neither scan fails the job.

The `k8s-test` cluster pulls the image tag pinned in the manifests, not the image that the same
run built.

## Image tags

The `build` job pushes to `docker.io/saidsef/ml-classifier`:

| Tag | Pushed by |
| --- | --- |
| `latest` | builds of `main` |
| `vYYYY.MM` | builds of `main`, overwritten by each build in that month |
| the last segment of the Git ref | every other build, such as `merge` for a pull request |

The manifests pin a `vYYYY.MM` tag, see [deploying on Kubernetes](deploying.md#image-tag).

## Tagging and releases

`.github/workflows/tagging.yml` runs after CI succeeds on `main`. It creates the Git tag
`vYYYY.MM` for the current month at the built commit, or moves the tag there if it exists.

`.github/workflows/release.yml` runs after tagging succeeds. It creates a GitHub release for that
tag with generated release notes. When the month's release exists, the workflow leaves it in
place.

## Documentation

Read the Docs builds this site from `mkdocs.yml` and `.readthedocs.yaml`. It installs the `docs`
dependency group from `uv.lock`, and MkDocs runs in strict mode, so a broken link fails the build.
To build the site locally:

```shell
uv run --only-group docs mkdocs serve
```
