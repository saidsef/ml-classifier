#!/bin/bash

set -ex

export TIMEOUT=${NOTEBOOK_TIMEOUT:-1200}

function runnb() {
  pip3 install $(cat ./requirements.txt | grep -i scikit-learn) -v
  time jupyter nbconvert --ExecutePreprocessor.timeout=$TIMEOUT --execute --to notebook --inplace classifier-ml.ipynb
}

runnb
