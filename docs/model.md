# Training the model

`classifier-ml.ipynb` turns the dataset into the model the service loads. CI runs it on every
build, and the same command runs it locally.

## Dataset

`data/news.json.xz` is an xz-compressed CSV file, despite its extension, stored in Git LFS. Each
row is one article:

| Column | Content |
| --- | --- |
| `body` | the article text, which is the only feature the model reads |
| `pubdate` | the publication timestamp |
| `subject` | the headline |
| `language` | the article language |
| `categories` | a comma-separated list of categories |

The notebook treats the whole `categories` string as one label, so
`Afghanistan, Asia & Americas, Europe & Russia` is a class of its own. The service returns these
strings as its categories.

## Preparation

The notebook drops duplicate rows, drops every category with fewer than ten articles and drops
rows with missing values. It splits the remainder 70/30 into training and test sets, stratified by
category.

## Pipeline

The model is a scikit-learn `Pipeline` of four steps:

| Step | Estimator | Role |
| --- | --- | --- |
| `vect` | `TfidfVectorizer` | turns the text into TF-IDF weights over word n-grams, without English stop words |
| `scaler` | `MaxAbsScaler` | scales each feature into the range -1 to 1 and keeps the matrix sparse |
| `feature_union` | `SelectKBest` and `KMeans` | keeps the features with the highest ANOVA F-score and appends the distance to each k-means centre |
| `clf` | `VotingClassifier` | averages the class probabilities of four estimators, with soft voting |

The four estimators are `LogisticRegression`, `RandomForestClassifier`, `MLPClassifier` and
`DecisionTreeClassifier`, each wrapped in a `SelfTrainingClassifier`. Soft voting gives the
pipeline a `predict_proba` method, which the service uses for the `score` in its responses. The
notebook cells hold the parameters of each step.

## Evaluation

The notebook prints the accuracy and a per-category classification report on the test set. The
committed notebook keeps the output of its last run.

## Output

After the evaluation, the notebook pickles the fitted pipeline with protocol 5 and compresses it with xz into
`data/voting_classifier.pickle.xz`. Git ignores the file. CI writes a SHA-256 checksum beside it,
and the image carries both files.

scikit-learn does not guarantee that a pickle loads under a different scikit-learn version. CI
installs the versions in `uv.lock` before it trains, and the image installs the same versions, so
the two always match. A change to `uv.lock` triggers CI, which retrains the model.

## Running the notebook

`scripts/run.sh` runs the notebook the way CI does. It installs uv, installs the locked runtime
dependencies into the container's Python, and executes the notebook in place. The Jupyter data
science image supplies pandas, seaborn and the other libraries the notebook imports beyond the
runtime dependencies.

```shell
git lfs pull
docker run --rm -v "$PWD":/app:rw -w /app -u root \
  quay.io/jupyter/datascience-notebook:latest bash ./scripts/run.sh
```

`NOTEBOOK_TIMEOUT` sets the time limit for each cell in seconds and defaults to 1600. The run
writes `data/voting_classifier.pickle.xz` and updates the outputs saved in the notebook.

The dev container under `.devcontainer/` builds on the same Jupyter image and forwards port 8888
for interactive work. Install the locked dependencies there before training, as `scripts/run.sh`
does, so that the model matches the scikit-learn version in the service image.
