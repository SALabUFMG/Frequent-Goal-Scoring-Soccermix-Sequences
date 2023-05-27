# Library imports
import numpy as np
import pandas as pd
import d6tflow as d6t
from tqdm import tqdm
import sklearn.model_selection as ms
import sklearn.metrics as mt
import xgboost as xgb
from socceraction.atomic.vaep import formula as av

# Project imports
from data_processing import loaders as ld
from vaeps import features as ft
from vaeps import labels as lb
from vaeps import training as tr
from vaeps import predictions as pr

class ValuesAtomicVAEP(d6t.tasks.TaskCSVPandas):
    competition = d6t.Parameter()
    train_comps = d6t.ListParameter()

    def requires(self):
        return self.clone(ld.AtomicSPADLLoader), self.clone(pr.PredsAtomicVAEP)

    def run(self):
        actions = self.input()[0].load()
        preds = self.input()[1].load()

        values = av.value(actions, preds['scores'], preds['concedes'])

        values['action_id'] = preds['action_id']
        values = values[['action_id'] + list(values.columns)[:-1]]

        self.save(values)