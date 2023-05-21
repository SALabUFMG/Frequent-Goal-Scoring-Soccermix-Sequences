# Library imports
import numpy as np
import d6tflow as d6t
import luigi as lg
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Project imports
sys.path.insert(1, '/home/gabriel/ic-SALab/IC-SALab-main/soccer_mix')
import MixtureModels.loaders as ld
import KULeuven.mixture as mix
import KULeuven.visualize as viz
import MixtureModels.dataprocessing as dp
import MixtureModels.categoricalmodel as cm

class LocationExperimentsSoccerMix(d6t.tasks.TaskCache):
    def requires(self):
        return cm.CategoricalSoccerMix()

    def run(self):
        cat_weights = self.input().load()

        '''experiments = list([dict(name=name, n_components=i, distribution=mix.MultiGauss)
                            for name in set(cat_weights.columns)
                            for i in range(1, 10)])
        experiments += list([dict(name=name, n_components=i, distribution=mix.MultiGauss)
                             for name in ["pass", "dribble", "receival"]
                             for i in range(10, 30)])'''

        loc_ultimate_selection = [("pass", 16), ("cross", 4), ("take_on", 7), ("shot", 3), ("bad_touch", 6),
                                  ("receival", 15), ("tackle", 4), ("interception", 5), ("clearance", 2), ("aerial_duel", 7)]

        experiments = [dict(name=e[0], n_components=e[1], distribution=mix.MultiGauss) for e in loc_ultimate_selection]

        self.save(experiments)

class LearnLocationSoccerMix(d6t.tasks.TaskPickle):
    def requires(self):
        return {'actions': dp.PrepSoccerMix(),
                'cat_weights': cm.CategoricalSoccerMix(),
                'experiments': LocationExperimentsSoccerMix()}

    def run(self):
        actions = self.input()['actions'].load()
        cat_weights = self.input()['cat_weights'].load()
        experiments = self.input()['experiments'].load()
        actions = actions.loc[cat_weights.index]

        loc_candidates = []
        experiments = tqdm(experiments, desc="learning location models")
        for experiment in experiments:
            loc_candidate = mix.MixtureModel(n_components=experiment["n_components"], distribution=experiment["distribution"])
            loc_candidate = loc_candidate.fit(actions[['x', 'y']], weights=cat_weights[experiment["name"]].values)
            if loc_candidate:
                loc_candidate.name = experiment["name"]
                loc_candidate.solo_bic = np.log(loc_candidate.weight_total) * loc_candidate._n_parameters() - 2 * loc_candidate.loglikelihood
                loc_candidates.append(loc_candidate)

        self.save(loc_candidates)

class SelectLocationSoccerMix(d6t.tasks.TaskPickle):
    def requires(self):
        return LearnLocationSoccerMix()

    def run(self):
        loc_candidates = self.input().load()

        '''loc_ultimate_selection = [("pass", 16), ("cross", 4), ("take_on", 7), ("shot", 3), ("bad_touch", 6), ("dribble", 10),
                                  ("receival", 15), ("tackle", 4), ("interception", 5), ("clearance", 2), ("pressure", 7)]
        loc_models = [l for l in loc_candidates if ((l.name, l.n_components) in loc_ultimate_selection)]'''

        loc_ultimate_selection = dict.fromkeys(list(set(c.name for c in loc_candidates)))
        for action in loc_ultimate_selection.keys():
            loc_ultimate_selection[action] = min(list(set(c.solo_bic for c in loc_candidates if c.name == action)))

        loc_models = [l for l in loc_candidates if (l.name in loc_ultimate_selection and l.solo_bic == loc_ultimate_selection[l.name])]

        self.save(loc_models)

class LocationWeightsSoccerMix(d6t.tasks.TaskPickle):
    def requires(self):
        return {'loc_models': SelectLocationSoccerMix(),
                'actions': dp.PrepSoccerMix(),
                'cat_weights': cm.CategoricalSoccerMix()}

    def run(self):
        actions = self.input()['actions'].load()
        cat_weights = self.input()['cat_weights'].load()
        actions = actions.loc[cat_weights.index]
        loc_models = self.input()['loc_models'].load()

        loc_weights = mix.probabilities(loc_models, actions[["x", "y"]], cat_weights)
        loc_weights.index = cat_weights.index

        self.save(loc_weights)