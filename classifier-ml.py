#!/usr/bin/env python

import os
from classifier import Classifier
from json import loads, dumps
from flask import Flask, request, jsonify

clf   = Classifier().model()
PORT  = os.environ.get("PORT")
app   = Flask(__name__)

@app.route('/api/v1/news', methods=['POST'])
def handler():
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
