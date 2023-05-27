# Library imports
import numpy as np
import pandas as pd
import d6tflow as d6t
from tqdm import tqdm
import sklearn.model_selection as ms
import sklearn.metrics as mt
import xgboost as xgb

# Project imports
from data_processing import loaders as ld
from vaeps import features as ft
from vaeps import labels as lb

class XGBVAEP(d6t.tasks.TaskPickle):
    train_comps = d6t.ListParameter()

    def requires(self):
        reqs = {'features': {}, 'labels': {}}

        for l in self.train_comps:
            reqs['features'][l] = ft.FeaturesVAEP(competition=l)
            reqs['labels'][l] = lb.LabelsVAEP(competition=l)

        return reqs

    def run(self):
        X_train = []
        y_train = []
        for l in self.train_comps:
            X_train.append(self.input()['features'][l].load())
            y_train.append(self.input()['labels'][l].load())
        X_train = pd.concat(X_train).reset_index(drop=True)
        y_train = pd.concat(y_train).reset_index(drop=True)

        models = {}
        for m in ['scores', 'concedes']:
            models[m] = xgb.XGBClassifier(random_state=0, n_estimators=50, max_depth=3)

            print('training ' + m + ' model')
            models[m].fit(X_train, y_train[m])

            p = sum(y_train[m]) / len(y_train[m])
            base = [p] * len(y_train[m])
            y_train_pred = models[m].predict_proba(X_train)[:, 1]
            train_brier = mt.brier_score_loss(y_train[m], y_train_pred) / mt.brier_score_loss(y_train[m], base)

            print(m + ' train nbs: ' + str(train_brier))
            print()

        self.save(models)

class XGBAtomicVAEP(d6t.tasks.TaskPickle):
    train_comps = d6t.ListParameter()

    def requires(self):
        reqs = {'features': {}, 'labels': {}}

        for l in self.train_comps:
            reqs['features'][l] = ft.FeaturesAtomicVAEP(competition=l)
            reqs['labels'][l] = lb.LabelsAtomicVAEP(competition=l)

        return reqs

    def run(self):
        X_train = []
        y_train = []
        for l in self.train_comps:
            X_train.append(self.input()['features'][l].load())
            y_train.append(self.input()['labels'][l].load())
        X_train = pd.concat(X_train).reset_index(drop=True)
        y_train = pd.concat(y_train).reset_index(drop=True)
        X_train = X_train.drop('action_id', axis=1)
        y_train = y_train.drop('action_id', axis=1)

        models = {}
        for m in ['scores', 'concedes']:
            models[m] = xgb.XGBClassifier(random_state=0, n_estimators=50, max_depth=3)

            print('training ' + m + ' model')
            models[m].fit(X_train, y_train[m])

            p = sum(y_train[m]) / len(y_train[m])
            base = [p] * len(y_train[m])
            y_train_pred = models[m].predict_proba(X_train)[:, 1]
            train_brier = mt.brier_score_loss(y_train[m], y_train_pred) / mt.brier_score_loss(y_train[m], base)

            print(m + ' train nbs: ' + str(train_brier))
            print()

        self.save(models)