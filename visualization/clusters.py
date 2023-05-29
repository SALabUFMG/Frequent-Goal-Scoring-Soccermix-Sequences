import pandas as pd
import numpy as np
import d6tflow as d6t


class LoadClusters(d6t.tasks.TaskCSVPandas):

    def run(self):
        path = r"H:\Documentos\SaLab\Frequent-Goal-Scoring-Soccermix-Sequences\visualization\loc_clusters.csv"
        clusters = pd.read_csv(path)
        clusters = clusters.drop(['Unnamed: 0'], axis=1)
        clusters = clusters.rename(columns={'id': 'cluster_id'})
        self.save(clusters)
        
class LoadZones(d6t.tasks.TaskCSVPandas):

    def run(self):
        path = r"H:\Documentos\SaLab\Frequent-Goal-Scoring-Soccermix-Sequences\visualization\grid_cells.csv"
        zones = pd.read_csv(path)
        zones = zones.drop(['Unnamed: 0'], axis=1)
        zones = zones.rename(columns={'id': 'zone_id'})
        self.save(zones)
