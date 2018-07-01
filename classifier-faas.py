#!?usr/bin/env python

import sys
from classifier import Classifier
from json import loads, dumps
from flask import Flask, request, jsonify

def get_stdin():
  buf = ""
  for line in sys.stdin:
      buf = buf + line
  return buf

def handler(text):
  clf = Classifier().model()
  prediction = clf.predict([text])
  score = clf.score([text], prediction)
  p = {'score': score, 'category': prediction[0]}
  return dumps(p)

if __name__ =='__main__':
  text = get_stdin()
  if text == '' or len(text) < 1:
    raise Exception('No input text')

  print(handler(text))
