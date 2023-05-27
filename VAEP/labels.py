import pandas as pd
import numpy as np
from tqdm import tqdm
import d6tflow as d6t

from socceraction.vaep import labels as lab

from DataProcessing.loader import WyLoader

class labels_transform(d6t.tasks.TaskCSVPandas):
    competition = d6t.Parameter()

    def requires(self):
        return WyLoader(competition=self.competition)

    def run(self):
        actions = self.inputLoad()
        yfns = [lab.scores, lab.concedes]

        labels = []
        for game in tqdm(np.unique(actions.game_id).tolist(), desc='Generating labels for {}'.format(self.competition)):
            match_actions = actions.loc[actions.game_id == game].reset_index(drop=True)
            labels.append(pd.concat([fn(actions=match_actions) for fn in yfns], axis=1))

        labels = pd.concat(labels).reset_index(drop=True)

        self.save(labels)