import pandas as pd
import numpy as np
from tqdm import tqdm
import d6tflow as d6t

import socceraction.vaep.formula as fm

from training import train_vaep
from features import features_transform
from loader import WyLoader

class generate_predictions():
    competition = d6t.Parameter()
    train_comps = d6t.ListParameter()

    def requires(self):
        return train_vaep(test_comp=self.competition, train_comps=self.train_comps), features_transform(competition=self.competition)

    def run(self):
        models = self.input()[0].load()
        features = self.input()[1].load()

        preds = {}
        for m in tqdm(['scores', 'concedes'], desc='Generating predictions'):
            preds[m] = models[m].predict_proba(features)[:, 1]
        preds = pd.DataFrame(preds)

        self.save(preds)

class calculate_action_values(d6t.tasks.TaskCSVPandas):
    competition = d6t.Parameter()
    train_comps = d6t.ListParameter()

    def requires(self):
        return generate_predictions(competition=self.competition, train_comps=self.train_comps), WyLoader(competition=self.competition)

    def run(self):
        predictions = self.inputLoad()
        actions = self.inputLoad()
        action_values = fm.value(actions=actions, Pscores=predictions['scores'], Pconcedes=predictions['concedes'])
        print(action_values.head(10))
        action_values = pd.concat([
            actions[['original_event_id','player_id','action_id', 'game_id', 'start_x', 'start_y', 'end_x', 'end_y', 'type_name', 'result_name']],
            predictions.rename(columns={'scores': 'Pscores', 'concedes': 'Pconcedes'}),
            action_values
        ], axis=1)

        self.save(action_values)