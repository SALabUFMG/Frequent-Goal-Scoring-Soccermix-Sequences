# Library imports
import os
import numpy as np
import pandas as pd
import d6tflow as d6t
from tqdm import tqdm
import ast
from pycspade.helpers import spade, print_result
from socceraction.vaep import features as vaep
from socceraction.atomic.vaep import features as avaep

# Project imports
from data_processing import loaders as ld
from ZoneSequences import actions_zones as az
from ZoneSequences import zone_sequences as zs
from vaeps import predictions as pr
from vaeps import action_values as av

@d6t.inherits(zs.ZoneSequences)
class MineZones(d6t.tasks.TaskCSVPandas):
    team_id = d6t.IntParameter()

    def requires(self):
        return {
            'actions': self.clone(ld.AtomicSPADLLoader),
            'zone_sequences': self.clone(zs.ZoneSequences)
        }

    def run(self):
        actions = self.input()['actions'].load()

        seq_actions = self.input()['zone_sequences']['actions'].load()
        actions = actions.merge(right=seq_actions, on='action_id')
        del seq_actions
        actions = actions[actions['team_id'] == self.team_id]

        sequences = self.input()['zone_sequences']['sequences'].load()
        sequences = sequences[sequences['sequence_id'].isin(actions['sequence_id'])]
        sequences['zones'] = sequences['zones'].apply(ast.literal_eval)

        # Check if directory exists
        dir_path = os.path.join(d6t.settings.dir, 'TeamZoneSequences')
        if not os.path.isdir(dir_path):
            # Directory doesn't exist, create it
            os.mkdir(dir_path)

        # Check if file exists within the directory
        file_path = os.path.join(dir_path, '{}.txt'.format(str(self.team_id)))
        if not os.path.isfile(file_path):
            seq_ids = sequences['sequence_id'].values.tolist()
            team_seqs = sequences['zones'].values.tolist()
            with open(file_path, 'w') as file:
                for i in range(len(team_seqs)):
                    seq_size = len(team_seqs[i])
                    for j in range(seq_size):
                        row = '{} {} 1 {}'.format(seq_ids[i], j + 1, team_seqs[i][j])
                        file.write(row + '\n')

        mined_seqs = spade(filename=file_path, support=0.001)['mined_objects']

        self.save({'actions': actions, 'sequences': sequences})

@d6t.inherits(zs.ZoneSequences)
class GridSeq2txt(d6t.tasks.TaskCache):
    team_id = d6t.IntParameter()
    pscores_threshold = d6t.FloatParameter()

    def requires(self):
        return {
            'actions': self.clone(ld.AtomicSPADLLoader),
            'zone_sequences': self.clone(zs.ZoneSequences),
            'vaep_preds': self.clone(pr.PredsAtomicVAEP, train_comps=['France', 'Germany', 'Italy']),
            'vaep_values': self.clone(av.ValuesAtomicVAEP, train_comps=['France', 'Germany', 'Italy'])
        }

    def run(self):
        actions = self.input()['actions'].load()

        seq_actions = self.input()['zone_sequences']['actions'].load()
        actions = actions.merge(right=seq_actions, on='action_id')
        del seq_actions
        actions = actions[actions['team_id'] == self.team_id]

        sequences = self.input()['zone_sequences']['sequences'].load()
        sequences = sequences[sequences['sequence_id'].isin(actions['sequence_id'])]
        sequences['zones'] = sequences['zones'].apply(ast.literal_eval)

        seq_actions = self.input()['zone_sequences']['actions'].load()
        seq_actions = seq_actions[seq_actions['sequence_id'].isin(actions['sequence_id'])]

        vaep_preds = self.input()['vaep_preds'].load()
        vaep_values = self.input()['vaep_values'].load()

        seq_actions = seq_actions.merge(right=vaep_preds, on='action_id')
        seq_actions = seq_actions.merge(right=vaep_values, on='action_id')

        seq_scores = seq_actions.groupby(by='sequence_id', as_index=False)['scores'].last()
        seq_values = seq_actions.groupby(by='sequence_id', as_index=False)['offensive_value'].sum()
        sequences = sequences.merge(right=seq_scores, on='sequence_id')
        sequences = sequences.merge(right=seq_values, on='sequence_id')

        sequences = sequences[sequences['scores'] >= self.pscores_threshold]

        # Check if directory exists
        dir_path = os.path.join(d6t.settings.dir, 'TeamZoneSequencesWagner_{}'.format(self.pscores_threshold))
        if not os.path.isdir(dir_path):
            # Directory doesn't exist, create it
            os.mkdir(dir_path)

        # Check if file exists within the directory
        file_path = os.path.join(dir_path, '{}.txt'.format(str(self.team_id)))
        if not os.path.isfile(file_path):
            team_seqs = sequences['zones'].values.tolist()
            scores = sequences['scores'].values.tolist()
            values = sequences['offensive_value'].values.tolist()
            with open(file_path, 'w') as file:
                for i in range(len(team_seqs)):
                    row = ''
                    row += ' -1 '.join(str(item) for item in team_seqs[i])
                    row += ' -1 -2 {} {}'.format(scores[i], values[i])
                    file.write(row + '\n')