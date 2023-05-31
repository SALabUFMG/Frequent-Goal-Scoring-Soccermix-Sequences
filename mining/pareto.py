# Library imports
import os
import pandas as pd
import d6tflow as d6t
from mining import soccermix as sm

@d6t.inherits(sm.SoccerMixSeq2txt)
class LoadParetoFrontiersSoccerMix(d6t.tasks.TaskCSVPandas):
    persist = ['sequences', 'instances']
    pareto = d6t.Parameter(default=' ')

    def run(self):
        if self.loc_and_dir:
            sm_type = 'loc_and_dir'
        else:
            sm_type = 'loc_only'

        directory = d6t.settings.dir + r'\pareto_soccermix_{}_{}\pareto{}'.format(sm_type, self.pscores_threshold, self.pareto)

        # Get all file names in the directory
        team_files = [f for f in os.listdir(directory) if str(self.team_id) in f and '.frontier' in f]

        seq_id = 1
        seq_ids = []
        frontiers = []
        sequences = []
        supports = []
        vaeps = []
        pscores = []
        lifts = []
        s_ids = []
        for i in range(len(team_files)):
            with open(directory + '\\' + team_files[i], 'r') as file:
                txt_file = file.read()
                lines = txt_file.split('\n')
                for l in lines[:-1]:
                    l = l.split(' ')
                    seq_ids.append(seq_id)
                    seq_id += 1
                    s_ids.append(int(l[0]))
                    frontiers.append(i + 1)
                    supports.append(int(l[3]))
                    vaeps.append(float(l[5]))
                    pscores.append(float(l[7]))
                    lifts.append(float(l[-1]))
                    seq = []
                    for j in range(8, len(l) - 4):
                        if l[j] != '-1':
                            seq.append(int(l[j]))
                    sequences.append(seq)

        sequences = pd.DataFrame(data={
            's_id': s_ids,
            'freq_seq_id': seq_ids,
            'frontier': frontiers,
            'sequence': sequences,
            'support': supports,
            'vaep': vaeps,
            'pscores': pscores,
            'lift': lifts
        })

        instances = sequences.explode('sequence').rename(columns={'sequence': 'item'}).drop(['support', 'vaep', 'lift', 'sids'], axis=1)
        instances['seq_order'] = instances.groupby(by='freq_seq_id')['freq_seq_id'].rank(method='first')

        self.save({'sequences': sequences, 'instances': instances})

@d6t.inherits(sm.SoccerMixSeq2txt)
class LoadParetoFrontiersZone(d6t.tasks.TaskCSVPandas):
    persist = ['sequences', 'instances']

    def run(self):
        directory = d6t.settings.dir + r'\pareto_zones_0.02\paretoC'

        # Get all file names in the directory
        team_files = [f for f in os.listdir(directory) if str(self.team_id) in f and '.frontier' in f]

        seq_id = 1
        seq_ids = []
        frontiers = []
        sequences = []
        supports = []
        vaeps = []
        pscores = []
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
                    pscores.append(float(l[7]))
                    lifts.append(float(l[-1]))
                    seq = []
                    for j in range(8, len(l) - 4):
                        if l[j] != '-1':
                            seq.append(int(l[j]))
                    sequences.append(seq)

        sequences = pd.DataFrame(data={
            'freq_seq_id': seq_ids,
            'frontier': frontiers,
            'sequence': sequences,
            'support': supports,
            'vaep': vaeps,
            'pscores': pscores,
            'lift': lifts
        })

        instances = sequences.explode('sequence').rename(columns={'sequence': 'item'}).drop(['support', 'vaep', 'lift'], axis=1)
        instances['seq_order'] = instances.groupby(by='freq_seq_id')['freq_seq_id'].rank(method='first')

        self.save({'sequences': sequences, 'instances': instances})
        
@d6t.inherits(sm.SoccerMixSeq2txt)
class LoadAllSoccermixSequences(d6t.tasks.TaskCSVPandas):
    persist = ['sequences', 'instances']
    
    def run(self):
        directory = d6t.settings.dir + r'\all_sequences'

        # Get all file names in the directory
        team_files = [f for f in os.listdir(directory) if str(self.team_id) in f and '.in' in f]

        seq_id = 1
        seq_ids = []
        frontiers = []
        sequences = []
        supports = []
        vaeps = []
        pscores = []
        lifts = []
        length = []
        for i in range(len(team_files)):
            with open(directory + '\\' + team_files[i], 'r') as file:
                txt_file = file.read()
                lines = txt_file.split('\n')
                
                for l in lines[:-1]:
                    l = l.split(' ')
                    seq_ids.append(seq_id)
                    seq_id += 1
                    frontiers.append(i + 1)
                    supports.append(int(l[2]))
                    vaeps.append(float(l[4]))
                    pscores.append(float(l[6]))
                    lifts.append(float(l[-1]))
                    length.append(len(l))
                    seq = []
                    for j in range(8, len(l) - 4):
                        if l[j] != '-1':
                            seq.append(int(l[j]))
                    sequences.append(seq)

        sequences = pd.DataFrame(data={
            'freq_seq_id': seq_ids,
            'frontier': frontiers,
            'sequence': sequences,
            'support': supports,
            'vaep': vaeps,
            'pscores': pscores,
            'lift': lifts,
            'length': length
        })

        instances = sequences.explode('sequence').rename(columns={'sequence': 'item'}).drop(['support', 'vaep', 'lift'], axis=1)
        instances['seq_order'] = instances.groupby(by='freq_seq_id')['freq_seq_id'].rank(method='first')

        self.save({'sequences': sequences, 'instances': instances})


@d6t.inherits(sm.SoccerMixSeq2txt)
class LoadSIDsSequences(d6t.tasks.TaskCSVPandas):
    persist = ['sequences', 'instances']

    def run(self):
        directory = d6t.settings.dir + r'\sequences_sid'

        # Get all file names in the directory
        team_files = [f for f in os.listdir(directory) if str(self.team_id) in f and '.outv' in f]

        seq_id = 1
        seq_ids = []
        frontiers = []
        sequences = []
        supports = []
        vaeps = []
        pscores = []
        lifts = []
        length = []
        sids = []
        for i in range(len(team_files)):
            with open(directory + '\\' + team_files[i], 'r') as file:
                txt_file = file.read()
                lines = txt_file.split('\n')

                for l in lines[:-1]:
                    sequencia = l.split('#SUP')[0].split(' ')[:-1]
                    data = l.split('#SUP: ')[1].split('#SID')[0].split(' ')
                    sid = l.split('#SID: ')[1].split(' ')
                    sid = [int(s) for s in sid if s != '']
                    sids.append(sid)
                    seq_ids.append(seq_id)
                    seq_id += 1
                    frontiers.append(i + 1)

                    supports.append(int(data[0]))
                    vaeps.append(float(data[2]))
                    pscores.append(float(data[4]))

                    seq = []
                    for s in sequencia:
                        if s != '-1':
                            seq.append(int(s))
                    sequences.append(seq)

        sequences = pd.DataFrame(data={
            'freq_seq_id': seq_ids,
            'frontier': frontiers,
            'sequence': sequences,
            'support': supports,
            'vaep': vaeps,
            'pscores': pscores,
            'sids': sids
        })

        instances = sequences.explode('sequence').rename(columns={'sequence': 'item'}).drop(['support', 'vaep'], axis=1)

        instances['seq_order'] = instances.groupby(by='freq_seq_id')['freq_seq_id'].rank(method='first')

        self.save({'sequences': sequences, 'instances': instances})