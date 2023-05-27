# Library imports
import numpy as np
import pandas as pd
from pandera.typing import DataFrame

def shoots(actions, nr_actions: int = 10):
    """Determine whether a shot was taken by the team possessing the ball within the next x actions.

        Parameters
        ----------
        actions : pd.DataFrame
            The actions of a game.
        nr_actions : int (default = 10)
            Number of actions after the current action to consider.

        Returns
        -------
        pd.DataFrame
            A dataframe with a column 'shoots' and a row for each action set to
            True if a shot was taken by the team possessing the ball within the
            next x actions; otherwise False.
        """
    # merging shots, penalties, free_kicks, and team_ids
    actions = actions.reset_index(drop=True)
    shots = ~actions.xg.isna()
    shot_idx = actions.loc[shots].index.values

    y = pd.concat([shots, actions['team_id']], axis=1)
    y.columns = ['shot', 'team']
    y = y.reset_index(drop=True)

    # adding future results
    for i in range(1, nr_actions):
        for c in ['shot', 'team']:
            shifted = y[c].shift(-i)
            shifted[-i:] = y[c][len(y) - 1]
            y['%s+%d' % (c, i)] = shifted

    res = np.ones(len(actions))
    res[shot_idx] = 1 - actions['xg'].values[shot_idx]
    for i in range(1, nr_actions):
        si = np.where(y['shot' + '+%d' % i] & (y['team+%d' % i] == y['team']))[0].tolist()
        si = np.array([v for v in si if v + i < len(actions)], dtype=int)
        res[si] *= 1 - actions['xg'].values[si + i]
    res = 1 - res

    res = pd.Series(data=res, name='scores')

    return res

def concedes(actions, nr_actions: int = 10):
    """Determine whether a shot was taken by the team possessing the ball within the next x actions.

        Parameters
        ----------
        actions : pd.DataFrame
            The actions of a game.
        nr_actions : int (default = 10)
            Number of actions after the current action to consider.

        Returns
        -------
        pd.DataFrame
            A dataframe with a column 'shoots' and a row for each action set to
            True if a shot was taken by the team possessing the ball within the
            next x actions; otherwise False.
        """
    # merging shots, penalties, free_kicks, and team_ids
    actions = actions.reset_index(drop=True)
    shots = ~actions.xg.isna()
    shot_idx = actions.loc[shots].index.values

    y = pd.concat([shots, actions['team_id']], axis=1)
    y.columns = ['shot', 'team']
    y = y.reset_index(drop=True)

    # adding future results
    for i in range(1, nr_actions):
        for c in ['team', 'shot']:
            shifted = y[c].shift(-i)
            shifted[-i:] = y[c][len(y) - 1]
            y['%s+%d' % (c, i)] = shifted

    res = np.ones(len(actions))
    for i in range(1, nr_actions):
        si = np.where(y['shot' + '+%d' % i] & (y['team+%d' % i] != y['team']))[0].tolist()
        si = np.array([v for v in si if v + i < len(actions)], dtype=int)
        res[si] *= 1 - actions['xg'].values[si + i]
    res = 1 - res

    res = pd.Series(data=res, name='concedes')

    return res