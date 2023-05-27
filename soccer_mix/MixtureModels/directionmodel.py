# Library imports
import numpy as np
import d6tflow as d6t
import luigi as lg
from tqdm import tqdm
import sys

# Project imports
#sys.path.insert(1, '/home/gabriel/ic-SALab/IC-SALab-main/soccer_mix')
import soccer_mix.KULeuven.mixture as mix
import soccer_mix.MixtureModels.dataprocessing as dp
import soccer_mix.MixtureModels.locationmodel as lm

@d6t.inherits(lm.LocationWeightsSoccerMix)
class DirectionExperimentsSoccerMix(d6t.tasks.TaskPickle):
    def requires(self):
        return self.clone(lm.LocationWeightsSoccerMix)

    def run(self):
        loc_weights = self.input().load()
        loc_weights = loc_weights.drop('action_id', axis=1)

        experiments = list([dict(name=name, n_components=i, distribution=mix.VonMises)
                            for name in set(loc_weights.columns)
                            for i in range(1, 5)])

        self.save(experiments)

@d6t.inherits(lm.LocationWeightsSoccerMix)
class LearnDirectionsSoccerMix(d6t.tasks.TaskPickle):
    def requires(self):
        return {
            'actions': self.clone(dp.PrepSoccerMix),
            'loc_weights': self.clone(lm.LocationWeightsSoccerMix),
            'experiments': self.clone(DirectionExperimentsSoccerMix)
        }

    def run(self):
        actions = self.input()['actions'].load()
        loc_weights = self.input()['loc_weights'].load()
        actions = actions.loc[actions['action_id'].isin(loc_weights['action_id'])]
        experiments = self.input()['experiments'].load()

        dir_candidates = []
        experiments = tqdm(experiments, desc="learning direction models")
        for experiment in experiments:
            dir_candidate = mix.MixtureModel(n_components=experiment["n_components"], distribution=experiment["distribution"])
            dir_candidate = dir_candidate.fit(actions[["mov_angle_a0"]], weights=loc_weights[experiment["name"]].values)
            dir_candidate.name = experiment["name"]
            dir_candidate.solo_bic = np.log(dir_candidate.weight_total) * dir_candidate._n_parameters() - 2 * dir_candidate.loglikelihood
            dir_candidates.append(dir_candidate)

        self.save(dir_candidates)

@d6t.inherits(LearnDirectionsSoccerMix)
class SelectDirectionExperimentsSoccerMix(d6t.tasks.TaskPickle):
    def requires(self):
        return {
            'dir_candidates': self.clone(LearnDirectionsSoccerMix)
        }

    def run(self):
        dir_candidates = self.input()['dir_candidates'].load()

        dir_ultimate_selection = dict.fromkeys(list(set(c.name for c in dir_candidates)))
        for action in dir_ultimate_selection.keys():
            dir_ultimate_selection[action] = min(list(set(c.solo_bic for c in dir_candidates if c.name == action)))

        dir_models = [l for l in dir_candidates if
                      (l.name in dir_ultimate_selection and l.solo_bic == dir_ultimate_selection[l.name])]

        self.save(dir_models)

@d6t.inherits(LearnDirectionsSoccerMix)
class DirectionWeightsSoccerMix(d6t.tasks.TaskPickle):
    def requires(self):
        return {
            'actions': self.clone(dp.PrepSoccerMix),
            'loc_weights': self.clone(lm.LocationWeightsSoccerMix),
            'dir_models': self.clone(SelectDirectionExperimentsSoccerMix)
        }

    def run(self):
        actions = self.input()['actions'].load()
        loc_weights = self.input()['loc_weights'].load()
        actions = actions.loc[actions['action_id'].isin(loc_weights['action_id'])]
        dir_models = self.input()['dir_models'].load()

        dir_weights = mix.probabilities(dir_models, actions[["mov_angle_a0"]], loc_weights.drop('action_id', axis=1))

        dir_weights['action_id'] = actions['action_id'].values
        dir_weights = dir_weights[['action_id'] + list(dir_weights.columns)[:-1]]

        self.save(dir_weights)