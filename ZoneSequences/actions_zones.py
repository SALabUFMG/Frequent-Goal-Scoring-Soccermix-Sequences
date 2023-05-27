# Library imports
import numpy as np
import pandas as pd
import d6tflow as d6t
from tqdm import tqdm
from socceraction.vaep import features as vaep
from socceraction.atomic.vaep import features as avaep

# Project imports
from data_processing import loaders as ld
from ZoneSequences import base

@d6t.inherits(ld.AtomicSPADLLoader)
class ActionsZones(d6t.tasks.TaskCSVPandas):
    def requires(self):
        return self.clone(ld.AtomicSPADLLoader)

    def run(self):
        actions = self.input().load()

        actions = actions[['action_id', 'x', 'y']]
        actions['zone'] = actions.apply(base.get_pitch_zone, axis=1)
        actions = actions.drop(['x', 'y'], axis=1)

        self.save(actions)