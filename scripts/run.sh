#!/bin/bash

set -ex

function runnb() {
  time jupyter nbconvert --ExecutePreprocessor.timeout=600 --execute --to notebook --inplace classifier-ml.ipynb
}

runnb
