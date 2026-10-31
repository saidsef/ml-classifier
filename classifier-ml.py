#!/usr/bin/env python3

import logging
from os import environ
from json import loads
from threading import Lock
from flask import Flask, request, jsonify, Response
from prometheus_flask_exporter import PrometheusMetrics
from classifier import Classifier

# Set up logging for the application
logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

PORT = environ.get("PORT", 7070)

# Initialize the classifier and Flask application
classifier = Classifier()
clf        = classifier.model()
app        = Flask(__name__)
training   = Lock()

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
  Response: A JSON response containing the predicted category and its probability as the score
  for POST requests, or a healthy message for GET requests.
  """
  if request.method == 'POST':
    data = loads(request.get_data())
    probabilities = clf.predict_proba([data['body']])[0]
    best = probabilities.argmax()
    p = {'score': round(float(probabilities[best]), 4), 'category': clf.classes_[best]}
    return jsonify(p)
  else:
    p = {'message': 'healthy'}
    return jsonify(p)

@app.route('/api/v1/train', methods=['POST'])
def train() -> tuple[Response, int]:
  """
  Endpoint to retrain the classifier on new data.

  Takes a JSON payload with 'body' and 'categories' keys, each mapping a document ID to its text
  and category, and replaces the served model with one fitted on that data alone.

  Returns:
  Response: A JSON response with the accuracy on the held-out documents, a 400 error if the data
  cannot be fitted, or a 409 error while another training run is in progress.
  """
  global clf
  if not training.acquire(blocking=False):
    return jsonify({'error': 'training already in progress'}), 409
  try:
    result = classifier.train(loads(request.get_data()))
    clf = classifier.model()
  except (KeyError, TypeError, ValueError) as e:
    return jsonify({'error': f'{type(e).__name__}: {e}'}), 400
  finally:
    training.release()
  return jsonify(result), 200

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=PORT)
