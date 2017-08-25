import unittest
import itertools
import json
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold

class TestDigits(unittest.TestCase):
    digits = load_digits(2)
    y = digits['target']
    X = digits['data']

    kf = KFold(n_splits=2, shuffle=True)
    for train_index, test_index in kf.split(X):
        xgb_model = xgb.XGBClassifier().fit(X[train_index],y[train_index])
        prediction = xgb_model.predict(X[test_index])
        actual = y[test_index]
    print(prediction)
    print(actual)
    assert prediction.all() == actual.all()
