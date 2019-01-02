#!/usr/bin/env python

import os
from classifier import Classifier
from json import loads, dumps
from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics

clf   = Classifier().model()
PORT  = os.environ.get("PORT")
app   = Flask(__name__)

#PrometheusMetrics(app, group_by='path')         # the default
#PrometheusMetrics(app, group_by='endpoint')     # by endpoint
PrometheusMetrics(app, group_by='url_rule')     # by URL rule

@app.route('/', methods=['GET'])
def index():
  return jsonify(['{} {}'.format(list(rule.methods), rule) for rule in app.url_map.iter_rules() if 'static' not in str(rule)])

@app.route('/api/v1/news', methods=['GET', 'POST'])
def handler():
  if request.method == 'POST':
    j = loads(request.get_data())
    prediction = clf.predict([j['payload']])
    score = clf.score([j['payload']], prediction)
    p = {'score': score, 'category': prediction[0]}
    return jsonify(p)
  else:
    p = {'message': 'healthy'}
    return jsonify(p)

if __name__ =='__main__':
  app.run(host='0.0.0.0', port=PORT)
