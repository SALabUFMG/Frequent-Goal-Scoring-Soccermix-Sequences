# Library imports
import numpy as np
import d6tflow as d6t
import pandas as pd
from tqdm import tqdm
import sys

# Project imports
#sys.path.insert(1, '/home/gabriel/ic-SALab/IC-SALab-main/soccer_mix')
import soccer_mix.KULeuven.mixture as mix
import soccer_mix.MixtureModels.dataprocessing as dp
import soccer_mix.MixtureModels.categoricalmodel as cm

@d6t.inherits(cm.CategoricalSoccerMix)
class LocationExperimentsSoccerMix(d6t.tasks.TaskCache):
    def requires(self):
        return self.clone(cm.CategoricalSoccerMix)

    def run(self):
        cat_weights = self.input().load().drop('action_id', axis=1)

        experiments = list([dict(name=name, n_components=i, distribution=mix.MultiGauss)
                            for name in set(cat_weights.columns)
                            for i in range(1, 10)])
        experiments += list([dict(name=name, n_components=i, distribution=mix.MultiGauss)
                             for name in ["pass", "dribble", "receival"]
                             for i in range(10, 30)])

        self.save(experiments)

@d6t.inherits(LocationExperimentsSoccerMix)
class LearnLocationsSoccerMix(d6t.tasks.TaskPickle):
    def requires(self):
        return {'actions': self.clone(dp.PrepSoccerMix),
                'cat_weights': self.clone(cm.CategoricalSoccerMix),
                'experiments': self.clone(LocationExperimentsSoccerMix)}

    def run(self):
        actions = self.input()['actions'].load()
        cat_weights = self.input()['cat_weights'].load()
        experiments = self.input()['experiments'].load()
        actions = actions.loc[actions['action_id'].isin(cat_weights['action_id'])]

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

@d6t.inherits(LearnLocationsSoccerMix)
class SelectLocationSoccerMix(d6t.tasks.TaskPickle):
    def requires(self):
        return self.clone(LearnLocationsSoccerMix)

    def run(self):
        loc_candidates = self.input().load()

        loc_ultimate_selection = dict.fromkeys(list(set(c.name for c in loc_candidates)))
        for action in loc_ultimate_selection.keys():
            loc_ultimate_selection[action] = min(list(set(c.solo_bic for c in loc_candidates if c.name == action)))

        loc_models = [l for l in loc_candidates if (l.name in loc_ultimate_selection and l.solo_bic == loc_ultimate_selection[l.name])]

        self.save(loc_models)

@d6t.inherits(SelectLocationSoccerMix)
class LocationWeightsSoccerMix(d6t.tasks.TaskCSVPandas):
    def requires(self):
        return {
            'actions': self.clone(dp.PrepSoccerMix),
            'cat_weights': self.clone(cm.CategoricalSoccerMix),
            'loc_models': self.clone(SelectLocationSoccerMix),
        }

    def run(self):
        actions = self.input()['actions'].load()
        cat_weights = self.input()['cat_weights'].load()
        actions = actions.loc[actions['action_id'].isin(cat_weights['action_id'])]
        loc_models = self.input()['loc_models'].load()

        loc_weights = mix.probabilities(loc_models, actions[["x", "y"]], cat_weights.drop('action_id', axis=1))

        loc_weights['action_id'] = actions['action_id'].values
        loc_weights = loc_weights[['action_id'] + list(loc_weights.columns)[:-1]]

        self.save(loc_weights)