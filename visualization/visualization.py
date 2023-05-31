import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch


def VisualizeTeam(team: int, df: pd.DataFrame, pareto: str, param: str):
    pitch = Pitch(line_color='black', pitch_type='custom', pitch_width=68, pitch_length=105)
    fig, ax = pitch.draw(figsize=(10, 7))

    n_seqs = len(df.freq_seq_id.unique())
    # sequencia = str(df['sequence'].values[0])

    for i, action in df.iterrows():
        x = action['loc_x']
        y = action['loc_y']
        end_x = df.loc[i + 1, 'loc_x'] if i < len(df) - 1 else None
        end_y = df.loc[i + 1, 'loc_y'] if i < len(df) - 1 else None
        next_seq = df.loc[i + 1, 'freq_seq_id'] if i < len(df) - 1 else None

        pitch.annotate(action['name'], (x - 2, y + 1.5), fontsize=12, ax=ax)
        pitch.scatter(x, y, alpha=0.5, s=150, color="lightgrey", ax=ax)

        if (end_x is not None) and (end_y is not None) and next_seq == action['freq_seq_id']:
            if action['cluster_id']!=df.loc[i + 1,'cluster_id']:
                pitch.arrows(x, y, end_x, end_y, color="blue", ax=ax, width=1.25, zorder=i, alpha=0.5)
            else:
                pitch.arrows(x, y-1, end_x, end_y+1, color="blue", ax=ax, width=1.5, zorder=i, alpha=0.35)

    title = "{} sequences".format(str(team))
    fig.suptitle(title, fontsize=24)
    
    subtitle = "top {} sequências por {}".format(str(n_seqs), param)
    fig.text(0.35, 0.9, subtitle, fontweight="regular")

    plt.show()


def VisualizeCluster(df: pd.DataFrame):
    pitch = Pitch(line_color='black', pitch_type='custom', pitch_width=68, pitch_length=105)
    fig, ax = pitch.draw(figsize=(10, 7))

    for i, action in df.iterrows():
        x = action['grid_x']
        y = action['grid_y']
        pitch.annotate(action['zone_id'], (x - 2, y + 1.5), fontsize=12, ax=ax)
        pitch.scatter(x, y, alpha=0.5, s=150, color="lightgrey", ax=ax)

    fig.suptitle("Clusters", fontsize=24)

    plt.show()
