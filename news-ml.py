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

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
print(__doc__)

PORT = os.environ.get("PORT")
pkl  = open('./data/lsvc.pickle', 'rb')
clf  = pickle.load(pkl)

app = Flask(__name__)

@app.route('/api/v1/news', methods=['POST'])
def news():
  if request.method == 'POST':
    j = loads(request.get_data())
    prediction = clf.predict([j['payload']])
    score = clf.score([j['payload']], prediction)
    p = {'score': score, 'category': prediction[0]}
    return dumps(p)
  else:
    p = {'message': 'healthy'}
    return dumps(p)

if __name__ =='__main__':
  app.run(host='0.0.0.0', port=PORT)
