import unittest
import itertools
import json
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_digits
from sklearn.model_selection import 

class TestDigits(unittest.TestCase):
      digits = load_digits(2)
      y = digits['target']
      X = digits['data']

      kf = KFold(n_splits=2, shuffle=True, random_state=rng)
        for train_index, test_index in kf.split(X):
          xgb_model = xgb.XGBClassifier().fit(X[train_index],y[train_index])
          predictions = xgb_model.predict(X[test_index])
          actuals = y[test_index]
          print(confusion_matrix(actuals, predictions))

      assert (prediction-actuals) == 0 
