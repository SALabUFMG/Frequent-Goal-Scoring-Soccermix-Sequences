# Library imports
import numpy as np
import d6tflow as d6t
import luigi as lg
import pandas as pd
from tqdm import tqdm
import sys
import seaborn as sns
import matplotlib.pyplot as plt

# Project imports
# sys.path.insert(1, '/home/gabriel/ic-SALab/IC-SALab-main/soccer_mix')
from data_processing import loaders as ld
import soccer_mix.KULeuven.mixture as mix
import soccer_mix.MixtureModels.dataprocessing as dp
import soccer_mix.MixtureModels.locationmodel as lm
import soccer_mix.MixtureModels.directionmodel as dm

@d6t.inherits(dm.DirectionWeightsSoccerMix)
class ActionsClustersSoccerMix(d6t.tasks.TaskCSVPandas):
    loc_and_dir = d6t.BoolParameter()   # True = location & direction; False = location only

    def requires(self):
        reqs = {}

        if self.loc_and_dir:
            reqs['weights'] = self.clone(dm.DirectionWeightsSoccerMix)
        else:
            reqs['weights'] = self.clone(lm.LocationWeightsSoccerMix)

        return reqs

    def run(self):
        weights = self.input()['weights'].load()

        receival_cols = [c for c in list(weights.columns) if 'receival' in c]
        weights = weights.drop(receival_cols, axis=1)
        weights = weights[weights.drop('action_id', axis=1).sum(axis=1) != 0]

        props = weights.iloc[:, 1:].values
        sorted_props = np.sort(props, axis=1)[:, ::-1]
        sorted_props_idxs = np.argsort(props, axis=1)[:, ::-1] + 1

        cumulative_props = np.cumsum(sorted_props, axis=1)
        cumulative_props[:, 1:] = cumulative_props[:, :-1]
        cumulative_props[:, 0] = 0

        thrshold_clusters = np.where(cumulative_props <= 0.9, sorted_props_idxs, 0)
        mask = thrshold_clusters > 0

        cluster_assignments = []
        for i in range(len(weights)):
            cluster_assignments.append(thrshold_clusters[i, mask[i, :]].tolist())

        cluster_assignments = pd.DataFrame(data={
            'action_id': weights['action_id'].values.tolist(),
            'soccermix_clusters': cluster_assignments
        })

        self.save(cluster_assignments)