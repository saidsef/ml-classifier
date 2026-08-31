#!/bin/bash

set -ex

export TIMEOUT=${NOTEBOOK_TIMEOUT:-1600}

function runnb() {
  pip3 install pipenv
  pipenv sync --system
  time jupyter nbconvert --ExecutePreprocessor.timeout=$TIMEOUT --execute --to notebook --inplace classifier-ml.ipynb
}

runnb
