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

The model is a scikit-learn `Pipeline` of two steps:

| Step | Estimator | Role |
| --- | --- | --- |
| `vect` | `TfidfVectorizer` | turns the text into TF-IDF weights over single words, ignoring words in fewer than two articles or more than 90% of them |
| `clf` | `CalibratedClassifierCV` around `LinearSVC` | fits a linear SVM with balanced class weights and converts its decision scores into probabilities with sigmoid calibration over three folds |

The calibration gives the pipeline a `predict_proba` method, which the service uses for the
`score` in its responses. The notebook cells hold the parameters of each step, and every step
that draws random numbers has a fixed seed, so repeated runs produce the same model.

## Choosing the regularisation

`C` controls how closely the SVM fits the training data. The notebook runs 5-fold
cross-validation on the training set for several values of `C` and fits the final model with the
value that scores best. The test set plays no part in the choice, so its accuracy is an unbiased
estimate.

## Evaluation

The notebook prints the accuracy, the top-3 accuracy, the log loss and a per-category
classification report on the test set. It then reloads the saved model and checks that its
predictions match the fitted model. The committed notebook keeps the output of its last run.

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
