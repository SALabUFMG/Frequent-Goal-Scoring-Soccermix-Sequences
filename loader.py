import d6tflow as d6t
import pandas as pd
from tqdm import tqdm

import socceraction.spadl as spadl

class WyLoader(d6t.tasks.TaskCSVPandas):
    competition = d6t.ListParameter()
    path = d6t.Parameter(default='data/')

    def requires(self):
        return load_matches(self.path, self.competition), load_events(self.path), load_minutes_played_per_game(self.path)

    def run(self):
        events = self.input()['events'].load()
        matches = self.input()['matches'].load()

        actions = []
        game_ids = events.game_id.unique().tolist()
        for g in tqdm(game_ids, desc='Converting {} events to SPADL'.format(self.competition)):
            match_events = events.loc[events.game_id == g]
            match_home_id = matches.loc[(matches.matchId == g) & (matches.side == 'home'), 'teamId'].values[0]
            match_actions = spadl.wyscout.convert_to_actions(events=match_events, home_team_id=match_home_id)
            match_actions = spadl.play_left_to_right(actions=match_actions, home_team_id=match_home_id)
            match_actions = spadl.add_names(match_actions)
            actions.append(match_actions)
        actions = pd.concat(actions).reset_index(drop=True)
        self.save(actions)


class load_matches(d6t.tasks.TaskCSVPandas):
    path = d6t.Parameter()
    competition = d6t.Parameter()

    def run(self):
        matches = pd.read_json(path_or_buf=self.path)
        team_matches = []
        for i in tqdm(range(len(matches)), desc='Loading {} matches'.format(self.competition)):
            match = pd.DataFrame(matches.loc[i, 'teamsData']).T
            match['matchId'] = matches.loc[i, 'wyId']
            team_matches.append(match)
        team_matches = pd.concat(team_matches).reset_index(drop=True)

        self.save(team_matches) 

class load_players(d6t.tasks.TaskCSVPandas):
    path = d6t.Parameter()
    def run(self):
        players = pd.read_json(path_or_buf=self.path)
        players = players[['wyId', 'shortName']].rename(columns={'wyId': 'player_id', 'shortName': 'player_name'})
        players['player_name'] = players['player_name'].str.decode('unicode-escape')

        self.save(players) 

class load_events(d6t.tasks.TaskCSVPandas):
    path = d6t.Parameter()
    def run(self):
        events = pd.read_json(path_or_buf=self.path)
        events = events.rename(columns={
            'id': 'event_id',
            'eventId': 'type_id',
            'subEventId': 'subtype_id',
            'teamId': 'team_id',
            'playerId': 'player_id',
            'matchId': 'game_id'
        })
        events['milliseconds'] = events['eventSec'] * 1000
        events['period_id'] = events['matchPeriod'].replace({'1H': 1, '2H': 2})

        self.save(events) 

class load_minutes_played_per_game(d6t.tasks.TaskCSVPandas):
    path = d6t.Parameter()
    def run(self):
        minutes = pd.read_json(path_or_buf=self.path)
        minutes = minutes.rename(columns={
            'playerId': 'player_id',
            'matchId': 'game_id',
            'teamId': 'team_id',
            'minutesPlayed': 'minutes_played'
        })
        minutes = minutes.drop(['shortName', 'teamName', 'red_card'], axis=1)

        self.save(minutes)
