# Library imports
import d6tflow as d6t
import numpy as np
import pandas as pd
from socceraction.atomic.vaep import features as fs
import sys

# Project imports
#sys.path.insert(1, '/home/gabriel/ic-SALab/IC-SALab-main/soccer_mix/MixtureModels')
from data_processing import loaders as ld

@d6t.inherits(ld.AtomicSPADLLoader)
class PrepSoccerMix(d6t.tasks.TaskCSVPandas):
    def requires(self):
        return self.clone(ld.AtomicSPADLLoader)


    def run(self):
        actions = self.input().load()

        action_types = ['pass', 'cross', 'dribble', 'take_on', 'shot', 'receival']

        actions = actions.loc[actions.type_name.isin(action_types)].reset_index(drop=True)

        # Add noise
        actions["x"] = actions.x + np.random.normal(0, 1, len(actions))
        actions["y"] = actions.y + np.random.normal(0, 1, len(actions))
        # actions["dx"] = actions.dx + (actions.dx != 0).apply(int) * np.random.normal(0,1,len(actions))
        # actions["dy"] = actions.dy + (actions.dy != 0).apply(int) * np.random.normal(0,1,len(actions))
        actions["dx"] = actions.dx + np.random.normal(0, 1, len(actions))
        actions["dy"] = actions.dy + np.random.normal(0, 1, len(actions))

        actions = pd.concat([actions, fs.movement_polar(actions)], axis=1)

        actions = actions[['action_id', 'type_name', 'x', 'y', 'dx', 'dy', 'mov_d_a0', 'mov_angle_a0']]

        self.save(actions)
