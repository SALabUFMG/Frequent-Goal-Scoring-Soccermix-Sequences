# Library imports
import numpy as np
import pandas as pd
import d6tflow as d6t
from tqdm import tqdm
from socceraction.vaep import features as vaep
from socceraction.atomic.vaep import features as avaep

# Project imports
from data_processing import loaders as ld

class FeaturesVAEP(d6t.tasks.TaskCSVPandas):
    competition = d6t.Parameter()

    def requires(self):
        return ld.SPADLLoader(competition=self.competition)

    def run(self):
        actions = self.input().load()
        actions.loc[actions.result_id.isin([2, 3]), ['result_id']] = 0
        actions.loc[actions.result_name.isin(['offside', 'owngoal']), ['result_name']] = 'fail'

        xfns = [
            vaep.actiontype_onehot,
            vaep.bodypart_onehot,
            vaep.result_onehot,
            vaep.goalscore,
            vaep.startlocation,
            vaep.endlocation,
            vaep.team,
            vaep.time,
            vaep.time_delta
        ]

        features = []
        for game in tqdm(np.unique(actions.game_id).tolist()):
            match_actions = actions.loc[actions.game_id == game].reset_index(drop=True)
            match_states = vaep.gamestates(actions=match_actions)
            match_feats = pd.concat([fn(match_states) for fn in xfns], axis=1)
            features.append(match_feats)
        features = pd.concat(features).reset_index(drop=True)

        self.save(features)

class FeaturesAtomicVAEP(d6t.tasks.TaskCSVPandas):
    competition = d6t.Parameter()

    def requires(self):
        return ld.AtomicSPADLLoader(competition=self.competition)

    def run(self):
        actions = self.input().load()

        xfns = [
            avaep.actiontype_onehot,
            avaep.bodypart_onehot,
            avaep.goalscore,
            avaep.location,
            avaep.team,
            avaep.time,
            avaep.time_delta
        ]

        features = []
        for game in tqdm(np.unique(actions.game_id).tolist()):
            match_actions = actions.loc[actions.game_id == game].reset_index(drop=True)
            match_states = vaep.gamestates(actions=match_actions)
            match_feats = pd.concat([fn(match_states) for fn in xfns], axis=1)
            features.append(match_feats)
        features = pd.concat(features).reset_index(drop=True)

        features['action_id'] = actions['action_id']
        features = features[['action_id'] + list(features.columns)[:-1]]

        self.save(features)