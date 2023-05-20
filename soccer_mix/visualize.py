# Library imports
import d6tflow as d6t
import luigi as lg
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances

# Project imports
import GaloPy.StatsBomb.interfaces as itf
import GaloPy.StatsBomb.API.loaders as ld
import GaloPy.SoccerMix.MixtureModels.locationmodel as lm
import GaloPy.SoccerMix.MixtureModels.directionmodel as dm
import GaloPy.SoccerMix.ActionVectors.vectorization as vec
import GaloPy.SoccerMix.KULeuven.visualize as viz

@d6t.inherits(itf.CompetitionInstance)
class VectorViz(d6t.tasks.TaskCache):
    ext_comp_inst = lg.Parameter(default=None)  # List of tuples (competition_id, season_id)
    level = lg.Parameter()  # 'player', 'team', 'manager'
    v1_key = lg.Parameter()
    v2_key = lg.Parameter(default=None)
    save_plots = lg.BoolParameter(default=True)
    normalize = lg.BoolParameter(default=False)

    def requires(self):
        reqs = {'loc_models': lm.SelectLocationExperiments(competition_season_id=self.competition_season_id),
                'dir_models': dm.SelectDirectionExperiments(competition_season_id=self.competition_season_id),
                'vectors': {}}

        if self.level == 'player':
            reqs['vectors'][self.competition_season_id] = vec.PlayerVectors(competition_season_id=self.competition_season_id)
            if self.ext_comp_inst != None:
                for comp_inst in self.ext_comp_inst:
                    reqs['vectors'][comp_inst] = vec.PlayerVectors(competition_season_id=self.competition_season_id, ext_comp_inst=comp_inst)

        elif self.level == 'team':
            reqs['vectors'][self.competition_season_id] = vec.TeamVectors(competition_season_id=self.competition_season_id)
            if self.ext_comp_inst != None:
                for comp_inst in self.ext_comp_inst:
                    reqs['vectors'][comp_inst] = vec.TeamVectors(competition_season_id=self.competition_season_id, ext_comp_inst=comp_inst)

        elif self.level == 'manager':
            reqs['vectors'][self.competition_season_id] = vec.TeamManagerVectors(competition_season_id=self.competition_season_id)
            if self.ext_comp_inst != None:
                for comp_inst in self.ext_comp_inst:
                    reqs['vectors'][comp_inst] = vec.TeamManagerVectors(competition_season_id=self.competition_season_id, ext_comp_inst=comp_inst)

        return reqs

    def run(self):
        loc_models = self.input()['loc_models'].load()['loc_models']
        dir_models = self.input()['dir_models'].load()['dir_models']

        vectors = {}
        if self.level == 'player':
            vectors[self.competition_season_id] = self.input()['vectors'][self.competition_season_id].load()['player_vectors']
            if self.ext_comp_inst != None:
                for comp_inst in self.ext_comp_inst:
                    vectors[comp_inst] = self.input()['vectors'][comp_inst].load()['player_vectors']
        else:
            vectors[self.competition_season_id] = self.input()['vectors'][self.competition_season_id].load()
            if self.ext_comp_inst != None:
                for comp_inst in self.ext_comp_inst:
                    vectors[comp_inst] = self.input()['vectors'][comp_inst].load()
        vectors = pd.concat(vectors.values())

        v1 = vectors.loc[vectors.index == self.v1_key].squeeze()
        if self.v2_key != None:
            v2 = vectors.loc[vectors.index == self.v2_key].squeeze()
        del vectors

        if self.normalize == True:
            group_cols = [c.split('_')[0] for c in v1.index]
            v1_T = v1.groupby(by=group_cols).sum()
            if self.v2_key != None:
                v2_T = v2.groupby(by=group_cols).sum()

            action_types = ['bad', 'clearance', 'cross', 'dribble', 'interception', 'pass', 'pressure', 'receival', 'shot', 'tackle', 'take']

            for action in action_types:
                action_cols = [c for c in list(set(v1.index)) if action in c]
                v1.loc[v1.index.isin(action_cols)] = v1.loc[v1.index.isin(action_cols)] / v1_T.loc[action]
                if self.v2_key != None:
                    v2.loc[v2.index.isin(action_cols)] = v2.loc[v2.index.isin(action_cols)] / v2_T.loc[action]

            #vectors = pd.DataFrame(data=preprocessing.normalize(vectors), index=vectors.index, columns=vectors.columns)

        if self.v2_key == None:
            axs = viz.show_vector_proportions(loc_models, dir_models, v1, self.v1_key, save=self.save_plots)
            #axs = viz.show_vector_proportions(loc_models, dir_models, v1, "Mohamed '20 '21", save=self.save_plots)

        else:
            #axs = viz.show_component_differences(loc_models, dir_models, v1, v2, self.v1_key, self.v2_key, save=self.save_plots)
            axs = viz.show_component_differences(loc_models, dir_models, v1, v2, "Cuca '21", "Mohamed '19 a '21", save=self.save_plots)

        self.save(axs)