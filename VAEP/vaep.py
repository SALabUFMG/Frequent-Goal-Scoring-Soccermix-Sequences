from loader import *

competitions = ['England','Spain']

# Load data
actions = {}
for competition in competitions:
    loader = WyLoader(competition=competition)
    actions[competition] = loader.outputLoad()

