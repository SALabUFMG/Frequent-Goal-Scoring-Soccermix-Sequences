import d6tflow as d6t
from mining import pareto as par
from visualization import clusters as cl
from visualization import visualization as vis
from DataProcessing.loader import load_teams
from tqdm import tqdm
import pandas as pd
import ast

d6t.settings

d6t.run(load_teams())
teams = load_teams().outputLoad()

team_ids = [1613,1673,1659,1651,1646,1631,1633,1639,1644,1623,1625,1627,1624,1628,
            1619, 1612, 1610, 1611, 1609, 10531]

pareto = 'C'
'''
for team_id in team_ids:
    team = teams.loc[teams['team_id']==team_id].team_name.values[0]
    d6t.run(par.LoadAllSoccermixSequences(competition='England', loc_and_dir=False, team_id=team_id, pscores_threshold=0.02, pareto=pareto))
    sequences = par.LoadAllSoccermixSequences(competition='England', loc_and_dir=False, team_id=team_id, pscores_threshold=0.02, pareto=pareto).output()['sequences'].load()
    sequences['sequence'] = sequences['sequence'].apply(ast.literal_eval)
    sequences['len'] = sequences['sequence'].apply(len)
    sequences = sequences.loc[sequences['len']>1].reset_index(drop=True)
    instances = par.LoadAllSoccermixSequences(competition='England', loc_and_dir=False, team_id=team_id, pscores_threshold=0.02, pareto=pareto).output()['instances'].load()
    
    d6t.run(cl.LoadClusters())
    clusters = cl.LoadClusters().output().load()
    seqs = sequences.freq_seq_id.unique().tolist()
    instances = instances.rename(columns={'item': 'cluster_id'})
    instances = instances.loc[instances['freq_seq_id'].isin(seqs)]
    
    params = ['len','lift','support','pscores','vaep']
    
    for param in params:
        sequences = sequences.sort_values(by=param, ascending=False).reset_index(drop=True)
        seqs = sequences.freq_seq_id.unique().tolist()
        seqs = seqs[:10]
        
        df = instances.merge(clusters, on='cluster_id', how='left').merge(sequences, on='freq_seq_id', how='left')
        df = df.loc[df.freq_seq_id.isin(seqs)].reset_index(drop=True)
        vis.VisualizeTeam(team=team, df=df, pareto=pareto, param=param)
'''

team_id = 1612
team = teams.loc[teams['team_id']==team_id].team_name.values[0]
d6t.run(par.LoadAllSoccermixSequences(competition='England', loc_and_dir=False, team_id=team_id, pscores_threshold=0.02, pareto=pareto))
sequences = par.LoadAllSoccermixSequences(competition='England', loc_and_dir=False, team_id=team_id, pscores_threshold=0.02, pareto=pareto).output()['sequences'].load()
sequences['sequence'] = sequences['sequence'].apply(ast.literal_eval)
sequences['len'] = sequences['sequence'].apply(len)
sequences = sequences.loc[sequences['len']>1].reset_index(drop=True)
instances = par.LoadAllSoccermixSequences(competition='England', loc_and_dir=False, team_id=team_id, pscores_threshold=0.02, pareto=pareto).output()['instances'].load()

d6t.run(cl.LoadClusters())
clusters = cl.LoadClusters().output().load()
seqs = sequences.freq_seq_id.unique().tolist()
instances = instances.rename(columns={'item': 'cluster_id'})
instances = instances.loc[instances['freq_seq_id'].isin(seqs)]

param = 'pscores'

sequences = sequences.sort_values(by=param, ascending=False).reset_index(drop=True)
seqs = sequences.freq_seq_id.unique().tolist()
seqs = seqs[:10]

df = instances.merge(clusters, on='cluster_id', how='left').merge(sequences, on='freq_seq_id', how='left')
df = df.loc[df.freq_seq_id.isin(seqs)].reset_index(drop=True)

df.to_csv('sequences.csv')

vis.VisualizeTeam(team=team, df=df, pareto=pareto, param=param)

#vis.VisualizeCluster(clusters)
