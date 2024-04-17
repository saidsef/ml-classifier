#!/bin/bash

set -ex

export TIMEOUT=${NOTEBOOK_TIMEOUT:-1400}

function runnb() {
  pip3 install -r ./requirements.txt
  time jupyter nbconvert --ExecutePreprocessor.timeout=$TIMEOUT --execute --to notebook --inplace classifier-ml.ipynb
}

runnb
