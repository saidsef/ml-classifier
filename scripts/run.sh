#!/bin/bash

set -ex

export TIMEOUT=${NOTEBOOK_TIMEOUT:-1200}

function runnb() {
  time jupyter nbconvert --ExecutePreprocessor.timeout=$TIMEOUT --execute --to notebook --inplace classifier-ml.ipynb
}

runnb
