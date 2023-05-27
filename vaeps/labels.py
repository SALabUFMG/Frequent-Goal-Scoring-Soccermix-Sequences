# Library imports
import numpy as np
import pandas as pd
import d6tflow as d6t
from tqdm import tqdm
from socceraction.vaep import labels as vaep
from socceraction.atomic.vaep import labels as avaep

# Project imports
from data_processing import loaders as ld
from vaeps import labels_base as lb

class LabelsVAEP(d6t.tasks.TaskCSVPandas):
    competition = d6t.Parameter()

    def requires(self):
        return ld.SPADLLoader(competition=self.competition)

    def run(self):
        actions = self.input().load()

        yfns = [vaep.scores, vaep.concedes]

        labels = []
        for game in tqdm(np.unique(actions.game_id).tolist()):
            match_actions = actions.loc[actions.game_id == game].reset_index(drop=True)
            labels.append(pd.concat([fn(actions=match_actions) for fn in yfns], axis=1))

        labels = pd.concat(labels).reset_index(drop=True)

        self.save(labels)

class LabelsAtomicVAEP(d6t.tasks.TaskCSVPandas):
    competition = d6t.Parameter()

    def requires(self):
        return ld.AtomicSPADLLoader(competition=self.competition)

    def run(self):
        actions = self.input().load()

        yfns = [avaep.scores, avaep.concedes]

        labels = []
        for game in tqdm(np.unique(actions.game_id).tolist()):
            match_actions = actions.loc[actions.game_id == game].reset_index(drop=True)
            labels.append(pd.concat([fn(actions=match_actions) for fn in yfns], axis=1))

        labels = pd.concat(labels).reset_index(drop=True)

        labels['action_id'] = actions['action_id']
        labels = labels[['action_id'] + list(labels.columns)[:-1]]

        self.save(labels)