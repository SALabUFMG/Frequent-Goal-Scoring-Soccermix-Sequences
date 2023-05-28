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
from soccer_mix.MixtureModels import sequences as sq
from vaeps import predictions as pr
from vaeps import action_values as av
from mining import zone as zn
from mining import soccermix as sm

@d6t.inherits(sm.SoccerMixSeq2txt)
class LoadParetoFrontiersSoccerMix(d6t.tasks.TaskCSVPandas):
    persist = ['sequences', 'instances']

    def run(self):
        if self.loc_and_dir:
            sm_type = 'loc_and_dir'
        else:
            sm_type = 'loc_only'

        directory = d6t.settings.dir + r'\pareto_soccermix_{}_{}'.format(sm_type, self.pscores_threshold)

        # Get all file names in the directory
        team_files = [f for f in os.listdir(directory) if str(self.team_id) in f and '.frontier' in f]

        seq_id = 1
        seq_ids = []
        frontiers = []
        sequences = []
        supports = []
        vaeps = []
        lifts = []
        for i in range(len(team_files)):
            with open(directory + '\\' + team_files[i], 'r') as file:
                txt_file = file.read()
                lines = txt_file.split('\n')
                for l in lines[:-1]:
                    l = l.split(' ')
                    seq_ids.append(seq_id)
                    seq_id += 1
                    frontiers.append(i + 1)
                    supports.append(int(l[3]))
                    vaeps.append(float(l[5]))
                    lifts.append(float(l[-1]))
                    seq = []
                    for j in range(6, len(l) - 4):
                        if l[j] != '-1':
                            seq.append(int(l[j]))
                    sequences.append(seq)

        sequences = pd.DataFrame(data={
            'freq_seq_id': seq_ids,
            'frontier': frontiers,
            'sequence': sequences,
            'support': supports,
            'vaep': vaeps,
            'lift': lifts
        })

        instances = sequences.explode('sequence').rename(columns={'sequence': 'item'}).drop(['support', 'vaep', 'lift'], axis=1)
        instances['seq_order'] = instances.groupby(by='freq_seq_id')['freq_seq_id'].rank(method='first')

        self.save({'sequences': sequences, 'instances': instances})

@d6t.inherits(sm.SoccerMixSeq2txt)
class LoadParetoFrontiersZone(d6t.tasks.TaskCSVPandas):
    persist = ['sequences', 'instances']

    def run(self):
        directory = d6t.settings.dir + r'\pareto_zones_{}'.format(self.pscores_threshold)

        # Get all file names in the directory
        team_files = [f for f in os.listdir(directory) if str(self.team_id) in f and '.frontier' in f]

        seq_id = 1
        seq_ids = []
        frontiers = []
        sequences = []
        supports = []
        vaeps = []
        lifts = []
        for i in range(len(team_files)):
            with open(directory + '\\' + team_files[i], 'r') as file:
                txt_file = file.read()
                lines = txt_file.split('\n')
                for l in lines[:-1]:
                    l = l.split(' ')
                    seq_ids.append(seq_id)
                    seq_id += 1
                    frontiers.append(i + 1)
                    supports.append(int(l[3]))
                    vaeps.append(float(l[5]))
                    lifts.append(float(l[-1]))
                    seq = []
                    for j in range(6, len(l) - 4):
                        if l[j] != '-1':
                            seq.append(int(l[j]))
                    sequences.append(seq)

        sequences = pd.DataFrame(data={
            'freq_seq_id': seq_ids,
            'frontier': frontiers,
            'sequence': sequences,
            'support': supports,
            'vaep': vaeps,
            'lift': lifts
        })

        instances = sequences.explode('sequence').rename(columns={'sequence': 'item'}).drop(['support', 'vaep', 'lift'], axis=1)
        instances['seq_order'] = instances.groupby(by='freq_seq_id')['freq_seq_id'].rank(method='first')

        self.save({'sequences': sequences, 'instances': instances})