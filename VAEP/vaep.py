import sys
sys.path.append("H:\Documentos\SaLab\Frequent-Goal-Scoring-Soccermix-Sequences")
from loader import *

competitions = ['England','Spain']

# Load data
actions = {}
minutes = {}

for competition in competitions:
    loader_task = WyLoader(competition=competition)
    d6t.run(loader_task)
    actions[competition] = loader_task.outputLoad()
    minutes_task = load_minutes_played_per_game(competition=competition)
    d6t.run(minutes_task)
    minutes[competition] = minutes_task.outputLoad()
