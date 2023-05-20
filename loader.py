import d6tflow as d6t
import pandas as pd
from tqdm import tqdm
from ast import literal_eval

import socceraction.spadl as spadl

class WyLoader(d6t.tasks.TaskCSVPandas):
    competition = d6t.ListParameter()

    def requires(self):
        return load_matches(self.competition), load_events(self.competition)

    def run(self):
        matches = self.input()[0].load()
        events = self.input()[1].load()
        events['tags'] = events['tags'].apply(literal_eval)
        events['positions'] = events['positions'].apply(literal_eval)
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
    competition = d6t.Parameter()

    def run(self):
        matches_path = r"H:\Documentos\SaLab\Soccermatics\Wyscout Data\matches_{}.json".format(self.competition)
        matches = pd.read_json(path_or_buf=matches_path)
        team_matches = []
        for i in tqdm(range(len(matches)), desc='Loading {} matches'.format(self.competition)):
            match = pd.DataFrame(matches.loc[i, 'teamsData']).T
            match['matchId'] = matches.loc[i, 'wyId']
            team_matches.append(match)
        team_matches = pd.concat(team_matches).reset_index(drop=True)

        self.save(team_matches) 

class load_players(d6t.tasks.TaskCSVPandas):

    def run(self):
        players_path = r"H:\Documentos\SaLab\Soccermatics\Wyscout Data\players.json"
        players = pd.read_json(path_or_buf=players_path)
        players = players[['wyId', 'shortName']].rename(columns={'wyId': 'player_id', 'shortName': 'player_name'})
        players['player_name'] = players['player_name'].str.decode('unicode-escape')

        self.save(players) 

class load_teams(d6t.tasks.TaskCSVPandas):

    def run(self):
        teams_path = r"H:\Documentos\SaLab\Soccermatics\Wyscout Data\teams.json"
        teams = pd.read_json(path_or_buf=teams_path)
        teams = teams[['wyId', 'name']].rename(columns={'wyId': 'team_id', 'name': 'team_name'})
        teams['team_name'] = teams['team_name'].str.decode('unicode-escape')

        self.save(teams) 

class load_events(d6t.tasks.TaskCSVPandas):
    competition = d6t.Parameter()

    def run(self):
        events_path = r"H:\Documentos\SaLab\Soccermatics\Wyscout Data\events_{}.json".format(self.competition)
        events = pd.read_json(path_or_buf=events_path)
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
    competition = d6t.Parameter()

    def run(self):
        minutes_path = "H:\Documentos\SaLab\Soccermatics\Wyscout Data\minutes_played_per_game_{}.json".format(self.competition)
        minutes = pd.read_json(path_or_buf=minutes_path)
        minutes = minutes.rename(columns={
            'playerId': 'player_id',
            'matchId': 'game_id',
            'teamId': 'team_id',
            'minutesPlayed': 'minutes_played',
        })
        minutes = minutes.drop(['shortName', 'teamName', 'red_card'], axis=1)
        minutes['shortName'] = minutes['player_name'].str.decode('unicode-escape')
        self.save(minutes)
