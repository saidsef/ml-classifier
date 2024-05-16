#!/usr/bin/env python3

import lzma
import logging
from pickle import load
from sklearn.model_selection import train_test_split

logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


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
    Initialises the Classifier instance by attempting to load a pre-trained model from a file.
    If the model file cannot be loaded, an error is logged.
    """
    try:
      with lzma.open('./data/voting_classifier.pickle.xz', 'rb') as fh:
        self.clf = load(fh)
    except IOError:
      logging.error("Unable to load file")
    finally:
      logging.info("Done loading file")

  def model(self) -> object:
    """
    Provides access to the loaded VotingClassifier model.

    Returns:
    VotingClassifier: The loaded model.
    """
    return self.clf

  def train(self, data) -> object:
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
