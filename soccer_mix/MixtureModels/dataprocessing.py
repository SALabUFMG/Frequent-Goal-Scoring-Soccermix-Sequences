# Library imports
import d6tflow as d6t
import numpy as np
import pandas as pd
from socceraction.atomic.vaep import features as fs
import sys

# Project imports
sys.path.insert(1, '/home/gabriel/ic-SALab/IC-SALab-main/soccer_mix/MixtureModels')
import loaders as ld
@d6t.inherits(ld.AtomicSPADLLoader)
class PrepSoccerMix(d6t.tasks.TaskCache):
    def requires(self):
        return self.clone(ld.AtomicSPADLLoader)


    def run(self):
        actions = self.input().load()

        op_actions = ['pass', 'cross', 'take_on', 'shot', 'bad_touch', 'receival',
                      'aerial_duel', 'tackle', 'interception', 'clearance']

        actions = actions.loc[actions.type_name.isin(op_actions)].reset_index(drop=True)

        # Add noise
        actions["x"] = actions.x + np.random.normal(0, 1, len(actions))
        actions["y"] = actions.y + np.random.normal(0, 1, len(actions))
        # actions["dx"] = actions.dx + (actions.dx != 0).apply(int) * np.random.normal(0,1,len(actions))
        # actions["dy"] = actions.dy + (actions.dy != 0).apply(int) * np.random.normal(0,1,len(actions))
        actions["dx"] = actions.dx + np.random.normal(0, 1, len(actions))
        actions["dy"] = actions.dy + np.random.normal(0, 1, len(actions))

        actions = pd.concat([actions, fs.movement_polar(actions)], axis=1)

        self.save(actions)
