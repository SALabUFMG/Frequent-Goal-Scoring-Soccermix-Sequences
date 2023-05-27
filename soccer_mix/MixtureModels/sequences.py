# Library imports
import numpy as np
import d6tflow as d6t
import luigi as lg
from tqdm import tqdm
import pandas as pd
import ast
import sys

# Project imports
#sys.path.insert(1, '/home/gabriel/ic-SALab/IC-SALab-main/soccer_mix')
from data_processing import loaders as ld
import soccer_mix.KULeuven.mixture as mix
import soccer_mix.MixtureModels.dataprocessing as dp
import soccer_mix.MixtureModels.locationmodel as lm
import soccer_mix.MixtureModels.directionmodel as dm
import soccer_mix.MixtureModels.actions_clusters as ac

@d6t.inherits(ac.ActionsClustersSoccerMix)
class SoccerMixSequences(d6t.tasks.TaskCSVPandas):
    persist = ['actions', 'sequences']

    def requires(self):
        return {
            'actions': self.clone(ld.AtomicSPADLLoader),
            'clusters': self.clone(ac.ActionsClustersSoccerMix)
        }

    def run(self):
        actions = self.input()['actions'].load()

        mask = (actions['team_id'].shift(-1) != actions['team_id']) | (
                    actions['game_id'].shift(-1) != actions['game_id']) | \
               (actions['period_id'].shift(-1) != actions['period_id']) | (actions['type_name'] == 'shot')
        true_idxs = actions[mask].index.values.tolist()

        clusters = self.input()['clusters'].load()
        actions = actions.merge(right=clusters, on='action_id', how='left')
        del clusters
        actions = actions[~actions['soccermix_clusters'].isna()]
        # Convert the string lists to actual lists
        actions['soccermix_clusters'] = actions['soccermix_clusters'].apply(ast.literal_eval)

        actions['sequence_id'] = int(0)

        sequences = {
            'sequence_id': [],
            'action_ids': [],
            'soccermix_sequence': []
        }
        start_idx = 0
        seq_id = 1

        for i in tqdm(range(len(true_idxs))):
            seq_actions = actions.loc[start_idx: true_idxs[i]]
            if len(seq_actions) > 0:
                sequences['sequence_id'].append(seq_id)
                sequences['action_ids'].append(seq_actions['action_id'].values.tolist())
                sequences['soccermix_sequence'].append(seq_actions['soccermix_clusters'].values.tolist())
                actions.loc[start_idx: true_idxs[i], 'sequence_id'] = seq_id
                seq_id += 1
            start_idx = true_idxs[i] + 1

        sequences = pd.DataFrame(sequences)
        actions = actions[['action_id', 'sequence_id']]

        self.save({'actions': actions, 'sequences': sequences})