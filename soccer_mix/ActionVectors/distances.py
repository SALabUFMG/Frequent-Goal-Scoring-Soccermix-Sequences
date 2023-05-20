# Library imports
import d6tflow as d6t
import luigi as lg
import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn import preprocessing
import sys

# Project imports
sys.path.insert(1, '/home/gabriel/ic-SALab/IC-SALab-main/soccer_mix')
import MixtureModels.loaders as ld
import ActionVectors.vectorization as vec

@d6t.inherits(vec.SoccerMixVectors)
class DeAnonymization(d6t.tasks.TaskPickle):
    def requires(self):
        reqs = [
            (ld.PossessionsLoader(competition='PL'), ld.FramesLoader(competition='PL')),
            self.clone(vec.SoccerMixVectors)
        ]

        return reqs

    def run(self):
        inputs = self.input()[1].load()
        player_ids = inputs[0]
        vectors_1 = inputs[1]
        vectors_2 = inputs[2]

        if len(self.input()) > 2:
            inputs = self.input()[2].load()
            vectors_1_d = inputs[1]
            vectors_2_d = inputs[2]
            vectors_1 = np.concatenate((vectors_1, vectors_1_d), axis=1)
            vectors_2 = np.concatenate((vectors_2, vectors_2_d), axis=1)
            del vectors_1_d, vectors_2_d

        D = pairwise_distances(vectors_1, vectors_2, metric="manhattan")

        # sort each row
        # k_d = np.sort(D, axis=1)
        # sort each row and replace distances by index
        k_i = np.argsort(D, axis=1)
        # replace indices by player ids
        p_i = np.take(player_ids, k_i, axis=0)

        rs = np.argmax(np.array([p_i[i, :] == player_ids[i] for i in range(p_i.shape[0])]), axis=1)

        def mean_reciprocal_rank(rs):
            return np.mean(1. / (rs + 1))

        def top_k(rs, k):
            return (rs < k).sum() / len(rs)

        mrr = mean_reciprocal_rank(rs)
        top1 = top_k(rs, 1)
        top3 = top_k(rs, 3)
        top5 = top_k(rs, 5)
        top10 = top_k(rs, 10)

        stats = [{'Position': 'All', 'Top1': top1, 'Top3': top3, 'Top5': top5, 'Top10': top10, 'MRR': mrr}]

        matches = self.input()[0][0]['matches'].load()
        matches = matches.sort_values('schedule_start').reset_index(drop=True)

        first_matches = matches.loc[:189, 'match_id'].values.tolist()
        second_matches = matches.loc[190:, 'match_id'].values.tolist()

        players = self.input()[0][0]['players'].load()
        players = players[players['position'] != 'goalkeeper']

        first_players = players[players['match_id'].isin(first_matches)].groupby(by='player_id', as_index=False)[
            'min_played'].sum()
        second_players = players[players['match_id'].isin(second_matches)].groupby(by='player_id', as_index=False)[
            'min_played'].sum()
        first_players = first_players[first_players['min_played'] >= 500]
        second_players = second_players[second_players['min_played'] >= 500]
        player_ids = list(set(first_players['player_id']) & set(second_players['player_id']))
        player_ids.sort()

        pl_pos = self.input()[0][1]['frames_coords'].load()
        pl_pos = pl_pos[pl_pos['player_id'].isin(player_ids)]
        pl_pos['player_position'] = pl_pos['player_position'].replace(
            {
                'RWB': 'FB', 'LWB': 'FB',
                'LCB': 'CB', 'RCB': 'CB',
                'DM': 'CM', 'RM': 'CM', 'LM': 'CM', 'AM': 'CM',
                'RW': 'F', 'RF': 'F', 'LW': 'F', 'LF': 'F', 'CF': 'F'
            }
        )
        pl_pos = pl_pos.groupby(by=['player_id', 'player_position'], as_index=False).size().rename(
            columns={'size': 'n_frames'})
        pl_pos['max_frames'] = pl_pos.groupby('player_id')['n_frames'].transform('max')
        mask = pl_pos['n_frames'] == pl_pos['max_frames']
        pl_pos = pl_pos[mask].drop(columns='max_frames').reset_index(drop=True).drop_duplicates('player_id')

        n_pos = pl_pos.groupby(by='player_position', as_index=False).size().rename(columns={'size': 'n_players'})
        n_pos = n_pos.groupby(by='player_position', as_index=False)['n_players'].sum()
        # n_pos['order'] = np.array([8, 2, 10, 5, 3, 7, 11, 0, 1, 6, 9])
        # n_pos = n_pos.sort_values('order').reset_index(drop=True)

        pl_pos = pl_pos.merge(right=players.groupby(by='player_id', as_index=False)['player_name'].first(),
                              on='player_id')

        # replace indices by player names
        n_i = np.take(players['player_name'].values.tolist(), k_i, axis=0)

        for p in pl_pos['player_position'].unique():
            mask = pl_pos['player_position'] == p
            stats.append({'Position': p, 'Top1': top_k(rs[mask], 1), 'Top3': top_k(rs[mask], 3),
                          'Top5': top_k(rs[mask], 5), 'Top10': top_k(rs[mask], 10),
                          'MRR': mean_reciprocal_rank(rs[mask])})

        stats = pd.DataFrame(stats)

        self.save((p_i, stats))