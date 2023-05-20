import d6tflow as d6t
import pandas as pd
from tqdm import tqdm

class WyLoader(d6t.tasks.TaskCSVPandas):
    competitions = d6t.ListParameter()

    def run(self):
      print()  


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
