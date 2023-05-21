# Library imports

import numpy as np
import d6tflow as d6t
import luigi as lg
from tqdm import tqdm
import sys

# Project imports
sys.path.insert(1, '/home/gabriel/ic-SALab/IC-SALab-main/soccer_mix')
import KULeuven.mixture as mix
import MixtureModels.dataprocessing as dp
import MixtureModels.locationmodel as lm

class DirectionExperiments(d6t.tasks.TaskPickle):
    def requires(self):
        return lm.LocationWeightsSoccerMix()

    def run(self):
        loc_weights = self.input().load()

        experiments = list([dict(name=name, n_components=i, distribution=mix.VonMises)
                            for name in set(loc_weights.columns)
                            for i in range(1, 5)])

        self.save(experiments)

class LearnDirectionMixes(d6t.tasks.TaskPickle):
    def requires(self):
        return (
            dp.PrepSoccerMix(),
            lm.LocationWeightsSoccerMix(),
            DirectionExperiments()
        )

    def run(self):
        atomic_actions = self.input()[0].load()
        loc_weights = self.input()[1].load()
        experiments = self.input()[2].load()

        dir_candidates = []
        experiments = tqdm(experiments, desc="learning direction models")
        for experiment in experiments:
            dir_candidate = mix.MixtureModel(n_components=experiment["n_components"],
                                     distribution=experiment["distribution"])
            dir_candidate = dir_candidate.fit(atomic_actions.loc[loc_weights.index.values, ["mov_angle_a0"]], weights=loc_weights[experiment["name"]].values)
            dir_candidate.name = experiment["name"]
            dir_candidate.solo_bic = np.log(dir_candidate.weight_total) * dir_candidate._n_parameters() - 2 * dir_candidate.loglikelihood
            dir_candidates.append(dir_candidate)

        self.save(dir_candidates)

class SelectDirectionExperiments(d6t.tasks.TaskPickle):
    def requires(self):
        return (
            dp.PrepSoccerMix(),
            lm.LocationWeightsSoccerMix(),
            LearnDirectionMixes()
        )

    def run(self):
        atomic_actions = self.input()[0].load()
        loc_weights = self.input()[1].load()
        dir_candidates = self.input()[2].load()

        dir_ultimate_selection = dict.fromkeys(list(set(c.name for c in dir_candidates)))
        for action in dir_ultimate_selection.keys():
            dir_ultimate_selection[action] = min(list(set(c.solo_bic for c in dir_candidates if c.name == action)))

        dir_models = [l for l in dir_candidates if
                      (l.name in dir_ultimate_selection and l.solo_bic == dir_ultimate_selection[l.name])]

        dir_weights = mix.probabilities(dir_models, atomic_actions.loc[loc_weights.index.values, ["mov_angle_a0"]], loc_weights)
        dir_weights.index = loc_weights.index

        self.save({'dir_models': dir_models, 'dir_weights': dir_weights})