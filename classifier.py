#!/usr/bin/env python

import os
import logging
import numpy as np
from json import loads, dumps
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib

class Classifier(object):
  def __init__(self):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    with open('./data/lsvc.pickle', 'rb') as fh:
      self.clf  = joblib.load(fh)

  def model(self):
    return self.clf
