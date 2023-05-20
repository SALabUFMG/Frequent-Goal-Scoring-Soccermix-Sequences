import d6tflow as d6t
import sys
sys.path.append("H:\Documentos\SaLab\Frequent-Goal-Scoring-Soccermix-Sequences")
from loader import *
from features import *
from labels import *
from training import *
from predictions import *

competitions = ['England','Spain']

# Load data
actions = {}
minutes = {}

# Load data
for competition in competitions:
    loader_task = WyLoader(competition=competition)
    d6t.run(loader_task)
    actions[competition] = loader_task.outputLoad()
    minutes_task = load_minutes_played_per_game(competition=competition)
    d6t.run(minutes_task)
    minutes[competition] = minutes_task.outputLoad()
    labels_task = labels_transform(competition=competition)
    d6t.run(labels_task)
    labels[competition] = labels_task.outputLoad()
    features_task = features_transform(competition=competition)
    d6t.run(features_task)
    features[competition] = features_task.outputLoad()


vaep_task = calculate_action_values(test_comp='England', train_comps=['Spain'])
d6t.run(vaep_task)
vaep = vaep_task.outputLoad()
