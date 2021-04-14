#!/usr/bin/env python3

import os
import lzma
import logging
from json import dumps
from pickle import load
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

class Classifier(object):
  def __init__(self):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    try:
      with lzma.open('./data/randomforestclassifier.pickle.xz', 'rb') as fh:
        self.clf  = load(fh)
    except IOError:
      logging.error("Unable to load file")
    finally:
      logging.info("Done loading file")

  def model(self):
    return self.clf

  def train(self, data):
    p = {'message': 'healthy'}
    try:
      xtrain, xtest, ytrain, ytest = train_test_split(dict(data['body']), dict(data['categories']), test_size=0.2, random_state=0)
      self.clf.fit(xtrain, ytrain)
    except Exception as e:
      p = {'error': "{}".format(str(e)) }
    finally:
      return p
