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

en_vaep_task = calculate_action_values(competition='England', train_comps=train_comps)
d6t.run(en_vaep_task)
vaepEngland = en_vaep_task.outputLoad()

sp_vaep_task = calculate_action_values(competition='Spain', train_comps=train_comps)
d6t.run(sp_vaep_task)
vaepSpain = sp_vaep_task.outputLoad()

vaep = pd.concat([vaepEngland, vaepSpain],axis=0).reset_index(drop=True)

df = vaep.merge(players, on='player_id').merge(teams, on='team_id')
df.to_csv('vaepActions.csv',index=True)

minutes = pd.concat([minutes['England'],minutes['Spain']],axis=0).reset_index(drop=True)
minutes_per_player = minutes.groupby('player_id', as_index=False)['minutes_played'].sum()

player_ratings = df.groupby(by='player_id', as_index=False).agg({'vaep_value': 'sum', 'player_name':'count'}).rename(columns={'vaep_value': 'vaep_total','player_name':'count'})
player_ratings = player_ratings.merge(minutes_per_player, on=['player_id'], how='left')
player_ratings['vaep_p90'] = player_ratings['vaep_total'] / player_ratings['minutes_played'] * 90
player_ratings = player_ratings[player_ratings['minutes_played'] >= 600].sort_values(by='vaep_p90', ascending=False).reset_index(drop=True)
player_ratings = player_ratings.merge(players, on=['player_id'], how='left')
player_ratings = player_ratings[['player_id', 'player_name', 'minutes_played', 'count','vaep_total', 'vaep_p90']]

team_ratings = df.groupby(by='team_id', as_index=False).agg({'vaep_value': 'sum', 'team_name':'count'}).rename(columns={'vaep_value': 'vaep_total','team_name':'count'})
team_ratings = team_ratings.merge(teams, on=['team_id'], how='left')
team_ratings['vaep_per_action'] = team_ratings['vaep_total'] / team_ratings['count']
team_ratings = team_ratings.sort_values(by='vaep_per_action', ascending=False).reset_index(drop=True)

action_ratings = df.groupby(by='type_name', as_index=True).agg({'vaep_value': 'sum', 'type_name':'count'}).rename(columns={'vaep_value': 'vaep_total','type_name':'count'})
action_ratings['vaep_per_action'] = action_ratings['vaep_total'] / action_ratings['count']
action_ratings = action_ratings.sort_values(by='vaep_per_action', ascending=False).reset_index(drop=False)

