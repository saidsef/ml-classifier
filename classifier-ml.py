#!/usr/bin/env python3

import os
import logging
from json import loads
from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
from classifier import Classifier

logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

PORT = os.environ.get("PORT", 8080)

clf   = Classifier().model()
app   = Flask(__name__)

PrometheusMetrics(app, group_by='url_rule')    # by URL rule

@app.route('/', methods=['GET'])
def index() -> str:
  ''' Returns list of available endpoints '''
  return jsonify(['{} {}'.format(list(rule.methods), rule) for rule in app.url_map.iter_rules() if 'static' not in str(rule)])

@app.route('/api/v1/news', methods=['GET', 'POST'])
def handler() -> str:
  ''' Takes string and returns string classification '''
  if request.method == 'POST':
    data = loads(request.get_data())
    prediction = clf.predict([data['body']])
    score = clf.score([data['body']], prediction)
    p = {'score': score, 'category': prediction[0]}
    return jsonify(p)
  else:
    p = {'message': 'healthy'}
    return jsonify(p)

@app.route('/api/v1/train', methods=['POST'])
def train() -> str:
  ''' Takes string and loads to training data '''
  if request.method == 'POST':
    data = loads(request.get_data())
    return Classifier().train(data)
  else:
    p = {'message': 'healthy'}
    return jsonify(p)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=PORT)
