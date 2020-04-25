#!/usr/bin/env python3

import os
import logging
from pickle import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class Classifier(object):
  def __init__(self):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    with open('./data/randomforestclassifier.pickle', 'rb') as fh:
      self.clf  = load(fh)

  def model(self):
    return self.clf
