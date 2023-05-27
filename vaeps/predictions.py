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
from vaeps import training as tr

class PredsVAEP(d6t.tasks.TaskCSVPandas):
    competition = d6t.Parameter()
    train_comps = d6t.ListParameter()

    def requires(self):
        return ft.FeaturesVAEP(competition=self.competition), tr.XGBVAEP(train_comps=self.train_comps)

    def run(self):
        features = self.input()[0].load()
        models = self.input()[1].load()

        preds = {}
        for m in ['scores', 'concedes']:
            preds[m] = models[m].predict_proba(features)[:, 1]
        preds = pd.DataFrame(preds)

        self.save(preds)

class PredsAtomicVAEP(d6t.tasks.TaskCSVPandas):
    competition = d6t.Parameter()
    train_comps = d6t.ListParameter()

    def requires(self):
        return ft.FeaturesAtomicVAEP(competition=self.competition), tr.XGBAtomicVAEP(train_comps=self.train_comps)

    def run(self):
        features = self.input()[0].load()
        models = self.input()[1].load()

        preds = {}
        for m in ['scores', 'concedes']:
            preds[m] = models[m].predict_proba(features.drop('action_id', axis=1))[:, 1]
        preds = pd.DataFrame(preds)

        preds['action_id'] = features['action_id']
        preds = preds[['action_id'] + list(preds.columns)[:-1]]

        self.save(preds)