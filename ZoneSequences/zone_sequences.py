# Library imports
import numpy as np
import pandas as pd
import d6tflow as d6t
from tqdm import tqdm
from socceraction.vaep import features as vaep
from socceraction.atomic.vaep import features as avaep

# Project imports
from data_processing import loaders as ld
from ZoneSequences import actions_zones as az

@d6t.inherits(ld.AtomicSPADLLoader)
class ZoneSequences(d6t.tasks.TaskCSVPandas):
    action_types = d6t.ListParameter()

    persist = ['actions', 'sequences']

    def requires(self):
        return {
            'actions': self.clone(ld.AtomicSPADLLoader),
            'zones': self.clone(az.ActionsZones)
        }

    def run(self):
        actions = self.input()['actions'].load()
        zones = self.input()['zones'].load()

        actions = actions.merge(right=zones, on='action_id')
        del zones

        mask = (actions['team_id'].shift(-1) != actions['team_id']) | (actions['game_id'].shift(-1) != actions['game_id']) | \
               (actions['period_id'].shift(-1) != actions['period_id']) | (actions['type_name'] == 'shot')
        true_idxs = actions[mask].index.values.tolist()

        actions = actions.loc[actions['type_name'].isin(list(self.action_types) + ['shot']), ['action_id', 'zone']]
        actions['sequence_id'] = int(0)

        sequences = {
            'sequence_id': [],
            'action_ids': [],
            'zones': []
        }
        start_idx = 0
        seq_id = 1

        for i in tqdm(range(len(true_idxs))):
            seq_actions = actions.loc[start_idx: true_idxs[i]]
            if len(seq_actions) > 0:
                sequences['sequence_id'].append(seq_id)
                sequences['action_ids'].append(seq_actions['action_id'].values.tolist())
                sequences['zones'].append(seq_actions['zone'].values.tolist())
                actions.loc[start_idx: true_idxs[i], 'sequence_id'] = seq_id
                seq_id += 1
            start_idx = true_idxs[i] + 1

        sequences = pd.DataFrame(sequences)
        actions = actions.drop('zone', axis=1)

        self.save({'actions': actions, 'sequences': sequences})
