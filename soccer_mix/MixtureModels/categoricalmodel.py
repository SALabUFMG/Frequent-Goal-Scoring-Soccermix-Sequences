# Library imports
import d6tflow as d6t
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor as LOF
import sys

# Project imports
#sys.path.insert(1, '/home/gabriel/ic-SALab/IC-SALab-main/soccer_mix')
import soccer_mix.KULeuven.mixture as mix
import soccer_mix.MixtureModels.dataprocessing as dp

@d6t.inherits(dp.PrepSoccerMix)
class CategoricalSoccerMix(d6t.tasks.TaskCSVPandas):
    def requires(self):
        return self.clone(dp.PrepSoccerMix)

    def run(self):
        actions = self.input().load()

        cat_model = mix.CategoricalModel()
        cat_model.fit(actions["type_name"])
        cat_weights = cat_model.predict_proba(actions["type_name"])

        new_weights = actions[['action_id']]
        for column in cat_weights.columns:
            idx = cat_weights[column] > cat_model.tol
            c = LOF(contamination="auto").fit_predict(actions[['x', 'y']][idx])
            idx[idx] = c == 1
            new_weights[column] = cat_weights[column].mask(~idx, 0)

        value_sums = new_weights.sum(axis=1) - new_weights['action_id']
        new_weights = new_weights.loc[value_sums == 1]

        self.save(new_weights)