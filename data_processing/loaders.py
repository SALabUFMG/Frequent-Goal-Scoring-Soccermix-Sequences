# Library imports
import numpy as np
import pandas as pd
import d6tflow as d6t
import json
import os
from tqdm import tqdm
import socceraction as sa
from ast import literal_eval
from socceraction import spadl
from socceraction.atomic import spadl as aspadl

# Project imports

# Wyscout PATH
PATH = r"H:/Documentos/SaLab/Soccermatics/Wyscout Data"

class CompetitionsLoader(d6t.tasks.TaskCSVPandas):
    def run(self):
        competitions = pd.read_json(path_or_buf=PATH + '/competitions.json')

        self.save(competitions)

class TeamsLoader(d6t.tasks.TaskCSVPandas):
    def run(self):
        teams = pd.read_json(path_or_buf=PATH + '/teams.json')

        self.save(teams)

class PlayersLoader(d6t.tasks.TaskCSVPandas):
    def run(self):
        players = pd.read_json(path_or_buf=PATH + '/players.json')

        self.save(players)

class CoachesLoader(d6t.tasks.TaskCSVPandas):
    def run(self):
        coaches = pd.read_json(path_or_buf=PATH + '/coaches.json')

        self.save(coaches)

class MatchesLoader(d6t.tasks.TaskCSVPandas):
    competition = d6t.Parameter()  # 'England', 'France', 'Germany', 'Italy', 'Spain'

    persist = ['matches_general', 'matches_specs']

    def run(self):
        matches_general = pd.read_json(path_or_buf=PATH + '/matches/matches_' + self.competition + '.json')

        matches_specs = []
        for i in tqdm(range(len(matches_general))):
            match = pd.DataFrame(matches_general.loc[i, 'teamsData']).T
            match['matchId'] = matches_general.loc[i, 'wyId']
            matches_specs.append(match)

        matches_specs = pd.concat(matches_specs).reset_index(drop=True)

        self.save({'matches_general': matches_general, 'matches_specs': matches_specs})

class EventsLoader(d6t.tasks.TaskCSVPandas):
    competition = d6t.Parameter()  # 'England', 'France', 'Germany', 'Italy', 'Spain'

    def run(self):
        events = pd.read_json(path_or_buf=PATH + '/events/events_' + self.competition + '.json')

        self.save(events)

class SPADLLoader(d6t.tasks.TaskCSVPandas):
    competition = d6t.Parameter()  # 'England', 'France', 'Germany', 'Italy', 'Spain'

    def requires(self):
        reqs = {'matches': MatchesLoader(competition=self.competition), 'events': EventsLoader(competition=self.competition)}

        return reqs

    def run(self):
        matches_specs = self.input()['matches']['matches_specs'].load()
        events = self.input()['events'].load()
        events['tags'] = events['tags'].apply(literal_eval)
        events['positions'] = events['positions'].apply(literal_eval)
        events = events.rename(columns={'id': 'event_id', 'eventId': 'type_id', 'subEventId': 'subtype_id',
                                        'teamId': 'team_id', 'playerId': 'player_id', 'matchId': 'game_id'})
        events['milliseconds'] = events['eventSec'] * 1000
        events['period_id'] = events['matchPeriod'].replace({'1H': 1, '2H': 2})

        actions = []
        game_ids = events.game_id.unique().tolist()
        for g in tqdm(game_ids):
            match_events = events.loc[events.game_id == g]
            match_home_id = matches_specs.loc[(matches_specs.matchId == g) & (matches_specs.side == 'home'), 'teamId'].values[0]
            match_actions = spadl.wyscout.convert_to_actions(events=match_events, home_team_id=match_home_id)
            match_actions = spadl.play_left_to_right(actions=match_actions, home_team_id=match_home_id)
            match_actions = spadl.add_names(match_actions)
            actions.append(match_actions)

        actions = pd.concat(actions).reset_index(drop=True)

        self.save(actions)

class AtomicSPADLLoader(d6t.tasks.TaskCSVPandas):
    competition = d6t.Parameter()  # 'England', 'France', 'Germany', 'Italy', 'Spain'

    def requires(self):
        return SPADLLoader(competition=self.competition)

    def run(self):
        actions = self.input().load()

        atomic_actions = aspadl.convert_to_atomic(actions)
        atomic_actions = aspadl.add_names(atomic_actions)

        self.save(atomic_actions)