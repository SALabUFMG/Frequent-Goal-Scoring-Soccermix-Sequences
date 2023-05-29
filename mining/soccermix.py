# Library imports
import os
import numpy as np
import pandas as pd
import d6tflow as d6t
from tqdm import tqdm
import ast
#from pycspade.helpers import spade
from socceraction.vaep import features as vaep
from socceraction.atomic.vaep import features as avaep

# Project imports
from data_processing import loaders as ld
from soccer_mix.MixtureModels import sequences as sq
from vaeps import predictions as pr
from vaeps import action_values as av

@d6t.inherits(sq.SoccerMixSequences)
class MineSoccerMix(d6t.tasks.TaskPickle):
    team_id = d6t.IntParameter()

    def requires(self):
        return {
            'actions': self.clone(ld.AtomicSPADLLoader),
            'soccermix_sequences': self.clone(sq.SoccerMixSequences)
        }

    def run(self):
        actions = self.input()['actions'].load()

        seq_actions = self.input()['soccermix_sequences']['actions'].load()
        actions = actions.merge(right=seq_actions, on='action_id')
        del seq_actions
        actions = actions[actions['team_id'] == self.team_id]

        sequences = self.input()['soccermix_sequences']['sequences'].load()
        sequences = sequences[sequences['sequence_id'].isin(actions['sequence_id'])]
        sequences['soccermix_sequence'] = sequences['soccermix_sequence'].apply(ast.literal_eval)

        # Check if directory exists
        dir_path = os.path.join(d6t.settings.dir, 'TeamSoccerMixSequences')
        if not os.path.isdir(dir_path):
            # Directory doesn't exist, create it
            os.mkdir(dir_path)

        # Check if file exists within the directory
        file_path = os.path.join(dir_path, '{}.txt'.format(str(self.team_id)))
        if not os.path.isfile(file_path):
            seq_ids = sequences['sequence_id'].values.tolist()
            team_seqs = sequences['soccermix_sequence'].values.tolist()
            with open(file_path, 'w') as file:
                for i in range(len(team_seqs)):
                    seq_size = len(team_seqs[i])
                    for j in range(seq_size):
                        row = '{} {} {} '.format(seq_ids[i], j + 1, len(team_seqs[i][j]))
                        row += ' '.join(str(item) for item in team_seqs[i][j])
                        file.write(row + '\n')

        #mined_seqs = spade(filename=file_path, support=0.01)['mined_objects']

        self.save(mined_seqs)

@d6t.inherits(sq.SoccerMixSequences)
class SoccerMixSeq2txt(d6t.tasks.TaskCache):
    team_id = d6t.IntParameter()
    pscores_threshold = d6t.FloatParameter()

    def requires(self):
        return {
            'actions': self.clone(ld.AtomicSPADLLoader),
            'soccermix_sequences': self.clone(sq.SoccerMixSequences),
            'vaep_preds': self.clone(pr.PredsAtomicVAEP, train_comps=['France', 'Germany', 'Italy']),
            'vaep_values': self.clone(av.ValuesAtomicVAEP, train_comps=['France', 'Germany', 'Italy'])
        }

    def run(self):
        actions = self.input()['actions'].load()

        seq_actions = self.input()['soccermix_sequences']['actions'].load()
        actions = actions.merge(right=seq_actions, on='action_id')
        del seq_actions
        actions = actions[actions['team_id'] == self.team_id]

        sequences = self.input()['soccermix_sequences']['sequences'].load()
        sequences = sequences[sequences['sequence_id'].isin(actions['sequence_id'])]
        sequences['soccermix_sequence'] = sequences['soccermix_sequence'].apply(ast.literal_eval)

        seq_actions = self.input()['soccermix_sequences']['actions'].load()
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
        dir_path = os.path.join(d6t.settings.dir, 'TeamSoccerMixSequencesWagner_{}'.format(self.pscores_threshold))
        if not os.path.isdir(dir_path):
            # Directory doesn't exist, create it
            os.mkdir(dir_path)

        # Check if file exists within the directory
        file_path = os.path.join(dir_path, '{}.txt'.format(str(self.team_id)))
        if not os.path.isfile(file_path):
            team_seqs = sequences['soccermix_sequence'].values.tolist()
            scores = sequences['scores'].values.tolist()
            values = sequences['offensive_value'].values.tolist()
            with open(file_path, 'w') as file:
                for i in range(len(team_seqs)):
                    seq_size = len(team_seqs[i])
                    row = ''
                    for j in range(seq_size):
                        row += ' '.join(str(item) for item in team_seqs[i][j])
                        row += ' -1 '
                    row += '-2 {} {}'.format(scores[i], values[i])
                    file.write(row + '\n')