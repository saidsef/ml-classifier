#!/bin/bash

set -exo pipefail

export TIMEOUT=${NOTEBOOK_TIMEOUT:-1600}

function runnb() {
  pip3 install uv
  uv export --frozen --no-dev | uv pip install --system -r -
  time jupyter nbconvert --ExecutePreprocessor.timeout=$TIMEOUT --execute --to notebook --inplace classifier-ml.ipynb
}

runnb
