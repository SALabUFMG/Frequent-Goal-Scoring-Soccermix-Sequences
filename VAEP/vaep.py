import d6tflow as d6t
import sys
sys.path.append("H:\Documentos\SaLab\Frequent-Goal-Scoring-Soccermix-Sequences")
from loader import *
from features import *
from labels import *
from training import *
from predictions import *

competitions = ['England','Spain']
train_comps = ['Italy','France','Germany']

# Load data
actions = {}
minutes = {}
labels = {}
features = {}

# Load data
for competition in competitions:
    loader_task = WyLoader(competition=competition)
    d6t.run(loader_task)
    actions[competition] = loader_task.outputLoad()
    minutes_task = load_minutes_played_per_game(competition=competition)
    d6t.run(minutes_task)
    minutes[competition] = minutes_task.outputLoad()
    labels_task = labels_transform(competition=competition)
    d6t.run(labels_task)
    labels[competition] = labels_task.outputLoad()
    features_task = features_transform(competition=competition)
    d6t.run(features_task)
    features[competition] = features_task.outputLoad()

players_task = load_players()
d6t.run(players_task)
players = players_task.outputLoad()

teams_task = load_teams()
d6t.run(teams_task)
teams = teams_task.outputLoad()

vaep_task = calculate_action_values(competition='England', train_comps=['Spain'])
d6t.run(vaep_task)
vaep = vaep_task.outputLoad()

df = vaep.merge(players, on='player_id').merge(teams, on='team_id')

minutes_per_player = minutes['England'].groupby('player_id', as_index=False)['minutes_played'].sum()

player_ratings = df.groupby(by='player_id', as_index=False).agg({'vaep_value': 'sum'}).rename(columns={'vaep_value': 'vaep_total'})
player_ratings = player_ratings.merge(minutes_per_player, on=['player_id'], how='left')
player_ratings['vaep_p90'] = player_ratings['vaep_total'] / player_ratings['minutes_played'] * 90
player_ratings = player_ratings[player_ratings['minutes_played'] >= 600].sort_values(by='vaep_p90', ascending=False).reset_index(drop=True)
player_ratings = player_ratings.merge(players, on=['player_id'], how='left')
player_ratings = player_ratings[['player_id', 'player_name', 'minutes_played', 'vaep_total', 'vaep_p90']]


