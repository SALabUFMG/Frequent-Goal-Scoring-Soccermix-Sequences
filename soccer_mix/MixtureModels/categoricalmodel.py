# Library imports
import d6tflow as d6t
import luigi as lg
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor as LOF
import sys
# Project imports

sys.path.insert(1, '/home/gabriel/ic-SALab/IC-SALab-main/soccer_mix')
import MixtureModels.loaders as ld
import KULeuven.mixture as mix
import MixtureModels.dataprocessing as dp

@d6t.inherits(ld.AtomicSPADLLoader)
class CategoricalSoccerMix(d6t.tasks.TaskCache):
    def requires(self):
        return self.clone(dp.PrepSoccerMix)

    def run(self):
        actions = self.input().load()

        cat_model = mix.CategoricalModel()
        cat_model.fit(actions["type_name"])
        cat_weights = cat_model.predict_proba(actions["type_name"])

        new_weights = pd.DataFrame()
        for column in cat_weights.columns:
            idx = cat_weights[column] > cat_model.tol
            c = LOF(contamination="auto").fit_predict(actions[['x', 'y']][idx])
            idx[idx] = c == 1
            new_weights[column] = cat_weights[column].mask(~idx, 0)

        value_sums = new_weights.sum(axis=1)
        new_weights = new_weights.loc[value_sums == 1]

        self.save(new_weights)