# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:07:22 2023

@author: jllgo
"""

import d6tflow as d6t
from soccer_mix.MixtureModels import sequences as sq
import mining.soccermix as sm
import ast
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from visualization import clusters as cl
from mining import pareto as par
import math
from matplotlib.patches import FancyArrowPatch
from DataProcessing.loader import load_teams

d6t.settings


pareto = 'C'

d6t.run(load_teams())
teams = load_teams().outputLoad()

team_ids = [1613,1673,1659,1651,1646,1631,1633,1639,1644,1623,1625,1627,1624,1628,
            1619, 1612, 1610, 1611, 1609, 10531]

for team_id in team_ids:
    d6t.run(par.LoadParetoFrontiersSoccerMix(competition='England', loc_and_dir=False, team_id=team_id, pscores_threshold=0.02, pareto=pareto))
    pareto_sequences = par.LoadParetoFrontiersSoccerMix(competition='England', loc_and_dir=False, team_id=team_id, pscores_threshold=0.02, pareto=pareto).output()['sequences'].load()
    pareto_sequences['sequence'] = pareto_sequences['sequence'].apply(ast.literal_eval)
    pareto_sequences['len'] = pareto_sequences['sequence'].apply(len)
    pareto_sequences = pareto_sequences.loc[pareto_sequences['len']>1].reset_index(drop=True)
    pareto_sequences = pareto_sequences.sort_values(by='len', ascending=False).reset_index(drop=True)
    s_ids = pareto_sequences.s_id.unique().tolist()
    
    d6t.run(par.LoadSIDsSequences(competition='England', loc_and_dir=False, team_id=team_id, pscores_threshold=0.02, pareto=pareto))
    sequences_sids = par.LoadSIDsSequences(competition='England', loc_and_dir=False, team_id=team_id, pscores_threshold=0.02, pareto=pareto).output()['sequences'].load()
    sequences_sids['sequence'] = sequences_sids['sequence'].apply(ast.literal_eval)
    sequences_sids['sids'] = sequences_sids['sids'].apply(ast.literal_eval)
    instances = par.LoadSIDsSequences(competition='England', loc_and_dir=False, team_id=team_id, pscores_threshold=0.02, pareto=pareto).output()['instances'].load()
    
    sequences_sids = sequences_sids.loc[sequences_sids['freq_seq_id']==s_ids[0]].reset_index(drop=True)
    instances = instances.loc[instances['freq_seq_id']==s_ids[0]].reset_index(drop=True)
    instances = instances.rename(columns={'item': 'cluster_id'})
    
    s_ids = sequences_sids['sids'].values[0]
    
    d6t.run(sm.SoccerMixTeamSeqIDs(team_id=team_id,competition='England',loc_and_dir=False,pscores_threshold=0.02))
    de_para_txt_seq = sm.SoccerMixTeamSeqIDs(team_id=team_id,competition='England',loc_and_dir=False,pscores_threshold=0.02).outputLoad()
    
    de_para_txt_seq = de_para_txt_seq.loc[de_para_txt_seq['txt_id'].isin(s_ids)].reset_index(drop=True)
    sequence_ids = de_para_txt_seq['sequence_id'].tolist()
    
    d6t.run(sq.SoccerMixSequences(loc_and_dir=False, competition='England'))
    sequences = sq.SoccerMixSequences(loc_and_dir=False,competition='England').output()['sequences'].load()
    sequences = sequences.loc[sequences['sequence_id'].isin(sequence_ids)].reset_index(drop=True)
    sequences['soccermix_sequence'] = sequences['soccermix_sequence'].apply(ast.literal_eval)
    
    general_instances = sequences.explode('soccermix_sequence')
    general_instances = general_instances.drop('action_ids', axis=1)
    general_instances['cluster_id'] = general_instances['soccermix_sequence'].apply(lambda x: x[0] if len(x) > 0 else None)
    
    d6t.run(cl.LoadClusters())
    clusters = cl.LoadClusters().output().load()
    general_instances = general_instances.merge(clusters, on='cluster_id', how='left')
    instances = instances.merge(clusters, on='cluster_id', how='left')
    
    def plot_sequence_build(sequence_df, general_df, seq_id, sequencia, team):
        pitch = Pitch(line_color='black', pitch_type='custom', pitch_width=68, pitch_length=105)
        fig, ax = pitch.draw(figsize=(10, 7))
    
        for i, action in general_df.iterrows():
            x = action['loc_x']
            y = action['loc_y']
            end_x = general_df.loc[i + 1, 'loc_x'] if i < len(general_df) - 1 else None
            end_y = general_df.loc[i + 1, 'loc_y'] if i < len(general_df) - 1 else None
            next_seq = general_df.loc[i + 1, 'sequence_id'] if i < len(general_df) - 1 else None
            
            #pitch.annotate(action['name'], (x - 2, y + 1.5), fontsize=12, ax=ax)
            #pitch.scatter(x, y, alpha=0.05, s=200, color="lightgrey", ax=ax)
            
            if (end_x is not None) and (end_y is not None) and next_seq == action['sequence_id']:
                if action['cluster_id']!=general_df.loc[i + 1,'cluster_id']:
                    pitch.arrows(x, y, end_x, end_y, color="blue", ax=ax, width=1.25, alpha=0.045)
                else:
                    arrow = FancyArrowPatch((x-1, y), (end_x+1, end_y), arrowstyle='->', connectionstyle='arc3,rad=2.5',
                            mutation_scale=10, color="blue", zorder=i, alpha=0.045)
                    ax.add_patch(arrow)
        for i, action in sequence_df.iterrows():
            x = action['loc_x']
            y = action['loc_y']
            end_x = sequence_df.loc[i + 1, 'loc_x'] if i < len(sequence_df) - 1 else None
            end_y = sequence_df.loc[i + 1, 'loc_y'] if i < len(sequence_df) - 1 else None
    
            pitch.annotate(action['name'], (x - 2, y + 1.5), fontsize=12, ax=ax)
            pitch.scatter(x, y, alpha=1, s=200, color="darkgrey", ax=ax)
    
            if (end_x is not None) and (end_y is not None):
                if action['cluster_id'] != sequence_df.loc[i + 1, 'cluster_id']:
                    pitch.arrows(x, y, end_x, end_y, color="red", ax=ax, width=2.5, zorder=1000000, alpha=1)
                else:
                    arrow = FancyArrowPatch((x-1, y), (end_x+1, end_y), arrowstyle='->', connectionstyle='arc3,rad=2.5',
                            mutation_scale=25, color="red", zorder=1000000, alpha=1)
                    ax.add_patch(arrow)
                    
        subtitle = "seq_id: {}      sequencia: {}".format(str(seq_id), str(sequencia))
        fig.text(0.33, 0.05, subtitle, fontweight="regular")
        title = "{}".format(str(team))
        fig.suptitle(title, fontsize=24)          
        
        plt.show()
        
    s_id = sequences_sids['freq_seq_id'].values[0]
    sequencia = sequences_sids['sequence'].values[0]
    team = teams.loc[teams['team_id']==team_id].team_name.values[0]
    plot_sequence_build(instances, general_instances, s_id, sequencia, team)

