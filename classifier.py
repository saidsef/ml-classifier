#!/usr/bin/env python3

import lzma
import logging
from pathlib import Path
from typing import Union, Dict
from pickle import load
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score

logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

MODEL = Path(__file__).parent / 'data' / 'voting_classifier.pickle.xz'


class Classifier(object):
  """
  A classifier for categorising text data using a VotingClassifier.

  This class encapsulates the functionality to load a pre-trained VotingClassifier model,
  provide access to the model, and support for re-training the model with new data.

  Attributes:
  clf (VotingClassifier): The loaded VotingClassifier model.
  """

  def __init__(self) -> None:
    """
    Initialises the Classifier instance by loading the pre-trained model from the data directory
    next to this file.

    Raises:
    - FileNotFoundError: If the model file is missing, naming the file and how to build it.
    """
    try:
      with lzma.open(MODEL, 'rb') as fh:
        self.clf = load(fh)
    except FileNotFoundError as e:
      raise FileNotFoundError(f'{MODEL} is missing, build it with classifier-ml.ipynb') from e

  def model(self) -> VotingClassifier:
    """
    Provides access to the loaded VotingClassifier model.

    Returns:
    VotingClassifier: The loaded model.
    """
    return self.clf

  def train(self, data: Dict[str, dict]) -> Dict[str, Union[float, int]]:
    """
    Retrains the classifier on the provided data and replaces the loaded model.

    A copy of the model is fitted on 80% of the documents and scored on the rest. The loaded
    model is replaced only once that fit succeeds, so predictions never see a partly fitted model.

    Parameters:
    - data (dict): A dictionary with two keys, 'body' and 'categories', each mapping a document ID
    to the document text and to its category respectively.

    Returns:
    - dict: The accuracy on the held-out documents, and the number of documents used to train and test.

    Raises:
    - KeyError, TypeError, ValueError: If the data is malformed or too small to fit the pipeline.
    """
    ids = list(data['body'])
    xtrain, xtest, ytrain, ytest = train_test_split(
      [data['body'][i] for i in ids], [data['categories'][i] for i in ids], test_size=0.2, random_state=0)
    model = clone(self.clf).fit(xtrain, ytrain)
    accuracy = accuracy_score(ytest, model.predict(xtest))
    self.clf = model
    return {'accuracy': round(float(accuracy), 4), 'train': len(xtrain), 'test': len(xtest)}
