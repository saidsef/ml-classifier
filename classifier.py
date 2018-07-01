#!/usr/bin/env python

import os
import pickle
import logging
import numpy as np
from json import loads, dumps
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline

class Classifier(object):
  def __init__(self):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    self.pkl  = open('./data/lsvc.pickle', 'rb')
    self.clf  = pickle.load(self.pkl)

  def model(self):
    return self.clf
