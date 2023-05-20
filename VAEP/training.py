import numpy as np
import pandas as pd
import d6tflow as d6t
from tqdm import tqdm
import xgboost as xgb
import sklearn.metrics as mt

from features import features_transform
from labels import labels_transform

class train_vaep(d6t.tasks.TaskPickle):
    test_comp = d6t.Parameter()
    train_comps = d6t.ListParameter()

    def requires(self):
        return LoadTrainTestData(test_comp=self.test_comp, train_comps=self.train_comps)
    
    def run(self):
        X_train = self.input()['X_train'].load()
        y_train = self.input()['y_train'].load()
        X_test = self.input()['X_test'].load()
        y_test = self.input()['y_test'].load()

        models = {}
        for m in ['scores', 'concedes']:
            models[m] = xgb.XGBClassifier(random_state=0, n_estimators=50, max_depth=3)

            print('training ' + m + ' model')
            models[m].fit(X_train, y_train[m])

            p = sum(y_train[m]) / len(y_train[m])
            base = [p] * len(y_train[m])
            y_train_pred = models[m].predict_proba(X_train)[:, 1]
            train_brier = mt.brier_score_loss(y_train[m], y_train_pred) / mt.brier_score_loss(y_train[m], base)
            print(m + ' Train NBS: ' + str(train_brier))
            print()

            p = sum(y_test[m]) / len(y_test[m])
            base = [p] * len(y_test[m])
            y_test_pred = models[m].predict_proba(X_test)[:, 1]
            test_brier = mt.brier_score_loss(y_test[m], y_test_pred) / mt.brier_score_loss(y_test[m], base)
            print(m + ' Test NBS: ' + str(test_brier))
            print()

            print('----------------------------------------')

        self.save(models)

class LoadTrainTestData(d6t.tasks.TaskCSVPandas):
    test_comp = d6t.Parameter()
    train_comps = d6t.ListParameter()

    persist = ['X_train', 'y_train', 'X_test', 'y_test']

    def requires(self):
        test_features = {}
        test_labels = {}
        train_features = {}
        train_labels = {}

        test_features = features_transform(competition=self.test_comp)
        test_labels = labels_transform(competition=self.test_comp)
        for competition in self.train_comps:
            train_features[competition] = features_transform(competition=competition)
            train_labels[competition] = labels_transform(competition=competition)

        return {
            'test_features': test_features,
            'test_labels': test_labels,
            'train_features': train_features,
            'train_labels': train_labels
        }
    
    def run(self):
        X_test = self.input()['test_features'].load()
        y_test = self.input()['test_labels'].load()
        train_features = {}
        train_labels = {}

        for competition in self.train_comps:
            train_features[competition] = self.input()['train_features'][competition].load()
            train_labels[competition] = self.input()['train_labels'][competition].load()

        X_train = pd.concat(train_features).reset_index(drop=True)
        y_train = pd.concat(train_labels).reset_index(drop=True)

        self.save(X_train, y_train, X_test, y_test)
