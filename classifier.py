#!/usr/bin/env python3

import lzma
import logging
from pathlib import Path
from typing import Union, Dict
from pickle import load
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier

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

  def train(self, data: Dict[str, dict]) -> Union[object, Dict[str, str]]:
    """
    Trains the classifier with the provided data.

    This method attempts to split the provided data into training and testing sets,
    then fits the classifier with the training data. If an error occurs during this process, 
    it returns a dictionary containing the error message.

    Parameters:
    - data (dict): A dictionary containing the data to train the classifier. It should have two 
    keys: 'body' and 'categories'. 'body' should map to the features, and 'categories' should map to the labels.

    Returns:
    - object: On successful training, it returns the trained classifier object. If an error occurs, 
    it returns a dictionary with the key 'error' and the error message as its value.

    Raises:
    - Exception: If an error occurs during the training process, an exception is caught and its message 
    is returned in a dictionary.
    """
    try:
      xtrain, xtest, ytrain, ytest = train_test_split(
        dict(data['body']), dict(data['categories']), test_size=0.2, random_state=0)
      self.clf.fit(xtrain, ytrain)
    except Exception as e:
      return {'error': "{}".format(str(e))}
