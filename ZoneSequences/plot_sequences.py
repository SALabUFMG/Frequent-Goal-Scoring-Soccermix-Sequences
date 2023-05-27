"""
arquivo de entrada: df das sequencias gerado por sequences.py

plota as sequencias

"""

import pandas as pd
from mplsoccer import Pitch
import matplotlib.pyplot as plt

def Get_Zone_Centroid_Mean(zone, df):
    x = 0
    y = 0
    coords = df[['start_x', 'start_y', 'zone']]
    coords = coords.groupby(by=['zone']).mean()
    for i in range (28):
        if zone == i+1:
            x = coords.iloc[i].values[0]
            y = coords.iloc[i].values[1]
    return x, y

df_actions = pd.read_csv(r"C:\Users\joaom\Documents\IC\codes\final\df_actions.csv")
df_sequences = pd.read_csv(r"C:\Users\joaom\Documents\IC\codes\final\df_sequences.csv")

# plot zones com centroids
pitch = Pitch(pitch_type='uefa', pitch_length=105, pitch_width=68)
fig, ax = pitch.draw(figsize=(10, 7))

X = [20, 36.25, 52.5, 68.75, 85]
Y = [14, 25, 43, 54]
for x in X:
    linha = plt.axvline(x, 0, 1, linestyle="--", color='lightgray')
    ax.add_line(linha)
for y in Y:
    if y in [25, 43]:
        linha = plt.axhline(y, 0.21, 1, linestyle="--", color='lightgray')
        ax.add_line(linha)
    else:
        linha = plt.axhline(y, 0, 1, linestyle="--", color='lightgray')
        ax.add_line(linha)

possible_zones = [i for i in range(1, 29)]

for zone in possible_zones:
    x = Get_Zone_Centroid_Mean(zone, df_actions)[0]
    y = Get_Zone_Centroid_Mean(zone, df_actions)[1]
    circleSize=1
    shotCircle=plt.Circle((x,y),circleSize,color="navy")
    ax.add_patch(shotCircle)
    
plt.show()

# escolhe as sequencias a serem plotadas (no caso, as com top 3 vaep)

top3_sequences = df_sequences.sort_values(by=['vaep'], ascending=False)[:3]
top3_sequences = top3_sequences[['zone_sequence']]
top3 = []
for i in range(len(top3_sequences)):
    top3.append(list(map(int, top3_sequences.iloc[i].values[0].split(' '))))

# plot das sequencias

pitch = Pitch(pitch_type='uefa', pitch_length=105, pitch_width=68)
fig, ax = pitch.draw(figsize=(10, 7))
colors = ['blue', 'red', 'green']

X = [20, 36.25, 52.5, 68.75, 85]
Y = [14, 25, 43, 54]
for x in X:
    linha = plt.axvline(x, 0, 1, linestyle="--", color='lightgray')
    ax.add_line(linha)
for y in Y:
    if y in [25, 43]:
        linha = plt.axhline(y, 0.21, 1, linestyle="--", color='lightgray')
        ax.add_line(linha)
    else:
        linha = plt.axhline(y, 0, 1, linestyle="--", color='lightgray')
        ax.add_line(linha)

for sequence, color in zip(top3, colors):
    for i in range(len(sequence) - 1):
        x = Get_Zone_Centroid_Mean(sequence[i], df_actions)[0]
        y = Get_Zone_Centroid_Mean(sequence[i], df_actions)[1]
        dx = Get_Zone_Centroid_Mean(sequence[i+1], df_actions)[0] - x
        dy = Get_Zone_Centroid_Mean(sequence[i+1], df_actions)[1] - y
        arrow = plt.Arrow(x, y, dx, dy, width=1, color=color)
        ax.add_patch(arrow)
        
plt.show()
