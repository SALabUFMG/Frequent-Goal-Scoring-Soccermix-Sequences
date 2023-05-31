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
'''
for team_id in team_ids:
    team = teams.loc[teams['team_id']==team_id].team_name.values[0]
    d6t.run(par.LoadParetoFrontiersZone(competition='England', loc_and_dir=False, team_id=team_id, pscores_threshold=0.02))
    sequences = par.LoadParetoFrontiersZone(competition='England', loc_and_dir=False, team_id=team_id, pscores_threshold=0.02).output()['sequences'].load()
    sequences['sequence'] = sequences['sequence'].apply(ast.literal_eval)
    sequences['len'] = sequences['sequence'].apply(len)
    sequences = sequences.loc[sequences['len']>1].reset_index(drop=True)
    seqs = sequences.freq_seq_id.unique().tolist()
    instances = par.LoadParetoFrontiersZone(competition='England', loc_and_dir=False, team_id=team_id, pscores_threshold=0.02).output()['instances'].load()
    instances = instances.rename(columns={'item': 'zone_id'})
    instances = instances.loc[instances['freq_seq_id'].isin(seqs)]

    d6t.run(cl.LoadZones())
    zones = cl.LoadZones().output().load()

    df = instances.merge(zones, on='zone_id', how='left').merge(sequences, on='freq_seq_id', how='left')
    vis.VisualizeTeam(team=team, df=df, pareto='C')

#vis.VisualizeCluster(zones)


'''

team_id = 1625
team = teams.loc[teams['team_id']==team_id].team_name.values[0]
d6t.run(par.LoadParetoFrontiersZone(competition='England', loc_and_dir=False, team_id=team_id, pscores_threshold=0.02))
sequences = par.LoadParetoFrontiersZone(competition='England', loc_and_dir=False, team_id=team_id, pscores_threshold=0.02).output()['sequences'].load()
sequences['sequence'] = sequences['sequence'].apply(ast.literal_eval)
sequences['len'] = sequences['sequence'].apply(len)
sequences = sequences.loc[sequences['len']>1].reset_index(drop=True)
seqs = sequences.freq_seq_id.unique().tolist()
instances = par.LoadParetoFrontiersZone(competition='England', loc_and_dir=False, team_id=team_id, pscores_threshold=0.02).output()['instances'].load()
instances = instances.rename(columns={'item': 'zone_id'})
instances = instances.loc[instances['freq_seq_id'].isin(seqs)]


d6t.run(cl.LoadZones())
zones = cl.LoadZones().output().load()

df = instances.merge(zones, on='zone_id', how='left').merge(sequences, on='freq_seq_id', how='left')


vis.VisualizeTeam(team=team, df=df, pareto='C')

#vis.VisualizeCluster(clusters)
