#!/usr/bin/env python3

import logging
from os import environ
from json import loads
from flask import Flask, request, jsonify, Response
from prometheus_flask_exporter import PrometheusMetrics
from classifier import Classifier
from sklearn.metrics import accuracy_score

# Set up logging for the application
logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

# Retrieve the port number from the environment variable or set to default 8080
PORT = environ.get("PORT", 8080)

# Initialize the classifier and Flask application
clf   = Classifier().model()
app   = Flask(__name__)

# Set up Prometheus metrics for Flask, grouped by URL rule
PrometheusMetrics(app, group_by='url_rule')    # by URL rule

@app.route('/', methods=['GET'])
def index() -> Response:
  """
  Endpoint to list all available API endpoints.

  Returns:
  Response: A JSON response containing a list of available endpoints.
  """
  return jsonify(['{} {}'.format(list(rule.methods), rule) for rule in app.url_map.iter_rules() if 'static' not in str(rule)])

@app.route('/api/v1/news', methods=['GET', 'POST'])
def handler() -> Response:
  """
  Endpoint to classify news text. Supports GET and POST methods.

  - POST: Takes a JSON payload with a 'body' key containing the text to classify.
  - GET: Returns a default healthy message.

  Returns:
  Response: A JSON response containing the classification score and category for POST requests,
  or a healthy message for GET requests.
  """
  if request.method == 'POST':
    data = loads(request.get_data())
    prediction = clf.predict([data['body']])
    score = accuracy_score([data['body']], prediction)
    p = {'score': '{:.4f}'.format(score) if score > 0.0 else '1', 'category': prediction[0]}
    return jsonify(p)
  else:
    p = {'message': 'healthy'}
    return jsonify(p)

@app.route('/api/v1/train', methods=['POST'])
def train() -> Response:
  """
  Endpoint to add new training data to the classifier.

  Takes a JSON payload with training data and updates the classifier model.

  Returns:
  Response: A JSON response containing a message indicating the result of the training operation.
  """
  if request.method == 'POST':
    data = loads(request.get_data())
    return Classifier().train(data)
  else:
    p = {'message': 'healthy'}
    return jsonify(p)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=PORT)
