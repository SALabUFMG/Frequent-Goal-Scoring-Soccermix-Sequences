import d6tflow as d6t
from tqdm import tqdm

from vaeps import predictions as pr
from vaeps import action_values as av

from soccer_mix.MixtureModels import dataprocessing as dp
from soccer_mix.MixtureModels import categoricalmodel as cm
from soccer_mix.MixtureModels import locationmodel as lm
from soccer_mix.MixtureModels import directionmodel as dm
from soccer_mix.MixtureModels import actions_clusters as ac
from soccer_mix.MixtureModels import sequences as sms

from ZoneSequences import actions_zones as az
from ZoneSequences import zone_sequences as zs

from mining import zone
from mining import soccermix as sm
from mining import pareto as par

d6t.settings

#d6t.run(av.ValuesAtomicVAEP(competition='England', train_comps=['France', 'Germany', 'Italy']))

#d6t.run(sms.SoccerMixSequences(competition='England', loc_and_dir=False))

#d6t.run(par.LoadParetoFrontiersSoccerMix(competition='England', loc_and_dir=True, team_id=1625, pscores_threshold=0.02))

d6t.run(par.LoadParetoFrontiersZone(competition='England', action_types=['pass', 'cross', 'dribble', 'take_on', 'shot'], team_id=1625, pscores_threshold=0.02))

'''teams = [1609, 1631, 1625, 1651, 1646, 1610, 1628, 1673, 1623, 1639, 1611, 1633, 1613, 1624, 10531, 1619, 1644, 1612, 1627, 1659]
for t in tqdm(teams):
    d6t.run(par.LoadParetoFrontiersSoccerMix(competition='England', loc_and_dir=True, team_id=t, pscores_threshold=0.02))
    d6t.run(zone.GridSeq2txt(competition='England', action_types=['pass', 'cross', 'dribble', 'take_on', 'shot'], team_id=t, pscores_threshold=0.))'''