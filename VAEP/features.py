import d6tflow as d6t
from tqdm import tqdm
import pandas as pd
import numpy as np

from socceraction.vaep import features as ft

from loader import WyLoader

class features_transform(d6t.tasks.TaskCSVPandas):
    competition = d6t.Parameter()

    def requires(self):
        return WyLoader(competition=self.competition)

    def run(self):
        actions = self.inputLoad()
        #actions.loc[actions.result_id.isin([2, 3]), ['result_id']] = 0
        #actions.loc[actions.result_name.isin(['offside', 'owngoal']), ['result_name']] = 'fail'

        xfns = [
            ft.actiontype_onehot,
            ft.bodypart_onehot,
            ft.result_onehot,
            ft.goalscore,
            ft.startlocation,
            ft.endlocation,
            ft.movement,
            ft.space_delta,
            ft.startpolar,
            ft.endpolar,
            ft.team,
            ft.time,
            ft.time_delta
        ]

        features = []
        for game in tqdm(np.unique(actions.game_id).tolist(), desc='Generating features for {}'.format(self.competition)):
            match_actions = actions.loc[actions.game_id == game].reset_index(drop=True)
            match_states = ft.gamestates(actions=match_actions)
            match_feats = pd.concat([fn(match_states) for fn in xfns], axis=1)
            features.append(match_feats)
        features = pd.concat(features).reset_index(drop=True)

        self.save(features)