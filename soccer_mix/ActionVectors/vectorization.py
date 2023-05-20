# Library imports
import d6tflow as d6t
import luigi as lg
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys

# Project imports
sys.path.insert(1, '/home/gabriel/ic-SALab/IC-SALab-main/soccer_mix')

import MixtureModels.loaders as ld
import MixtureModels.dataprocessing as dp
import MixtureModels.categoricalmodel as cm
import MixtureModels.directionmodel as dm

class SoccerMixVectors(d6t.tasks.TaskPickle):
    aggr = d6t.Parameter()  # 'mean' or 'p90'

    def requires(self):
        return (
            ld.PossessionsLoader(competition='PL'),
            dp.PrepSoccerMix(),
            dm.SelectDirectionExperiments()
        )

    def run(self):
        matches = self.input()[0]['matches'].load()
        matches = matches.sort_values('schedule_start').reset_index(drop=True)

        first_matches = matches.loc[:189, 'match_id'].values.tolist()
        second_matches = matches.loc[190:, 'match_id'].values.tolist()

        players = self.input()[0]['players'].load()
        players = players[players['position'] != 'goalkeeper']

        first_players = players[players['match_id'].isin(first_matches)].groupby(by='player_id', as_index=False)[
            'min_played'].sum()
        second_players = players[players['match_id'].isin(second_matches)].groupby(by='player_id', as_index=False)[
            'min_played'].sum()
        first_players = first_players[first_players['min_played'] >= 500]
        second_players = second_players[second_players['min_played'] >= 500]
        player_ids = list(set(first_players['player_id']) & set(second_players['player_id']))
        player_ids.sort()
        first_players = first_players[first_players['player_id'].isin(player_ids)].sort_values('player_id').reset_index(
            drop=True)
        second_players = second_players[second_players['player_id'].isin(player_ids)].sort_values(
            'player_id').reset_index(drop=True)

        atomic_actions = self.input()[1].load()
        dir_weights = self.input()[2].load()['dir_weights']
        atomic_actions = atomic_actions[atomic_actions.index.isin(dir_weights.index)]

        vectors_1, vectors_2 = [], []
        for p in tqdm(player_ids, desc="building player vectors"):
            mask_1st = (atomic_actions['player_id'] == p) & (atomic_actions['match_id'].isin(first_matches))
            values_1st = dir_weights.loc[mask_1st].sum().values
            if self.aggr == 'mean':
                values_1st = values_1st / np.sum(values_1st)
            elif self.aggr == 'p90':
                values_1st = values_1st / first_players.loc[first_players['player_id'] == p, 'min_played'].max() * 90

            mask_2nd = (atomic_actions['player_id'] == p) & (atomic_actions['match_id'].isin(second_matches))
            values_2nd = dir_weights.loc[mask_2nd].sum().values
            if self.aggr == 'mean':
                values_2nd = values_2nd / np.sum(values_2nd)
            elif self.aggr == 'p90':
                values_2nd = values_2nd / second_players.loc[second_players['player_id'] == p, 'min_played'].max() * 90

            vectors_1.append(pd.DataFrame(data=[values_1st], columns=list(set(dir_weights.columns))))
            vectors_2.append(pd.DataFrame(data=[values_2nd], columns=list(set(dir_weights.columns))))

        # To DataFrame
        vectors_1 = pd.concat(vectors_1)
        vectors_2 = pd.concat(vectors_2)

        self.save((player_ids, vectors_1, vectors_2))