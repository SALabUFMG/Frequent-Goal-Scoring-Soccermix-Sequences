'''
arquivo de entrada: dataframe VAEP
    
gera um dataframe com todas as ações, contendo:
    - informações gerais da ação
    - número da sequencia
    - zonas que pertencem à sequencia de cada ação
    - eventId das ações que pertencem à sequencia de cada ação

gera um dataframe resumido das sequencias, contendo:
    - vaep da sequencia
    - zonas da sequencia
    - eventIds da sequencia

'''
import pandas as pd
import numpy as np
from tqdm import tqdm
from data_processing import loaders as ld

def Get_Pitch_Zone(x, y):
    if x < 20:
        if y <= 14:
            return 3
        elif y <= 54:
            return 2
        else:
            return 1
    elif x < 36.25:
        if y <= 14:
            return 8
        elif y <= 25:
            return 7
        elif y <= 43:
            return 6
        elif y <= 54:
            return 5
        else:
            return 4
    elif x < 52.5:
        if y <= 14:
            return 13
        elif y <= 25:
            return 12
        elif y <= 43:
            return 11
        elif y <= 54:
            return 10
        else:
            return 9
    elif x < 68.75:
        if y <= 14:
            return 18
        elif y <= 25:
            return 17
        elif y <= 43:
            return 16
        elif y <= 54:
            return 15
        else:
            return 14
    elif x < 85:
        if y <= 14:
            return 23
        elif y <= 25:
            return 22
        elif y <= 43:
            return 21
        elif y <= 54:
            return 20
        else:
            return 19
    else:
        if y <= 14:
            return 28
        elif y <= 25:
            return 27
        elif y <= 43:
            return 26
        elif y <= 54:
            return 25
        else:
            return 24
        
# ler o df de ações com vaep
#df = pd.read_csv(r"C:\Users\joaom\Documents\IC\vaep_England.csv")
df = ld.SPADLLoader(competition='England').output().load()
df = df.dropna(subset=['end_x', 'end_y', 'start_x', 'start_y'])
df = df.reset_index()

# calcula as zonas de cada ação / separa as ações em sequencias / soma o VAEP por sequencia
zones = [Get_Pitch_Zone(df['start_x'][0], df['start_y'][0])]
len_seq = 0
len_seqs = [] # vetor armazena o tamanho de cada sequencia
seq = 0
seqs = [0] # vetor com o numero da sequencia que cada ação pertence
#vaep_seq = [df['vaep_value'][0]]
vaeps_finais_long = []
vaeps_finais = []
times = []
for i in tqdm(range(1, len(df))):
    zones.append(Get_Pitch_Zone(df['start_x'][i], df['start_y'][i]))
    if (df['team_id'][i - 1] != df['team_id'][i]) or (df['game_id'][i - 1] != df['game_id'][i]) or (df['period_id'][i - 1] != df['period_id'][i]) or (df['result_name'][i-1] == 'fail') or (df['type_name'][i-1] == 'shot'):
        seq += 1
        len_seqs.append(len_seq + 1)
        '''vaep_seq.append(df['vaep_value'][i])
        if len_seq > 1:
            vaeps_finais_long.append(vaep_seq[i-1])
        vaeps_finais.append(vaep_seq[i-1])'''
        times.append(df['team_id'][i-1])
        len_seq = 0
    else:
        len_seq += 1
        #vaep_seq.append(vaep_seq[i-1] + df['vaep_value'][i])
    seqs.append(seq)
len_seqs.append(len_seq+1)

df['zone'] = zones
df['num_seq'] = seqs
df['vaep_acumulado'] = vaep_seq
del len_seq, seq, vaep_seq, seqs, zones

# pega todas as sequencias de zonas e eventIds
zone_sequences = np.array([], dtype=object)
event_sequences = []
event_sequence = []
num_seq = 0
zone = df['zone'][0]
sequencia = str(zone)
for i in range(1, len(df)):
    zone = df['zone'][i]
    if df['num_seq'][i] == df['num_seq'][i-1]:
        sequencia += ' ' + str(zone)
        event_sequence.append(df['original_event_id'][i])
    else:
        zone_sequences = np.insert(zone_sequences, num_seq, sequencia)
        num_seq += 1
        sequencia = str(zone)
        event_sequences.append(event_sequence)
        event_sequence = [df['original_event_id'][i]]
zone_sequences = np.insert(zone_sequences, num_seq, sequencia)
event_sequences.append(event_sequence)
del num_seq, sequencia, zone

# adiciona a sequencia da qual cada ação pertence ao dataframe
zone_sequences_ext = []
event_sequences_ext = []
k = 0
for i in range(len(zone_sequences)):
    for j in range(len_seqs[k]):
        zone_sequences_ext.append(zone_sequences[i])
        event_sequences_ext.append(event_sequences[i])
    k += 1
    
df['zone_sequence'] = zone_sequences_ext
df['event_sequence'] = event_sequences_ext
del i, j, k, zone_sequences_ext

# adiciona o tamanho de cada sequencia ao dataframe
len_seqs_aux = []
for i in range(len(len_seqs)):
    for j in range(len_seqs[i]):
        len_seqs_aux.append(len_seqs[i])
df['len_seq'] = len_seqs_aux

df_sequences = pd.DataFrame()
seqs_iteradas = 0
team = []
sequence = []
vaep = []
events = []

for i in range(1, len(df)):
    if df['num_seq'].iloc[i] != df['num_seq'].iloc[i-1]:
        team.append(df['team_name'].iloc[i-1])
        sequence.append(df['zone_sequence'].iloc[i-1])
        events.append(df['event_sequence'].iloc[i-1])
        vaep.append(df['vaep_acumulado'].iloc[i-1])
        seqs_iteradas += 1

df_sequences['team_name'] = team
df_sequences['zone_sequence'] = sequence
df_sequences['event_ids'] = events
df_sequences['vaep'] = vaep

# df.to_csv('df_actions.csv')
# df_sequences.to_csv('df_sequences.csv')
