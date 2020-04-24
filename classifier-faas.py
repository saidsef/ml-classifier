#!/usr/bin/env python3

import sys
import logging
from classifier import Classifier
from json import loads, dumps

logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

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
