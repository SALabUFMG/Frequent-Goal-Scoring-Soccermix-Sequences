import pandas as pd
import numpy as np
import re
from tqdm import tqdm


def readData(team_id):
    out_path = r"H:\Documentos\SaLab\Frequent-Goal-Scoring-Soccermix-Sequences\lift\zones\spmf_{}.out".format(team_id)
    in_path = r"H:\Documentos\SaLab\Frequent-Goal-Scoring-Soccermix-Sequences\lift\zones\df_{}.csv".format(team_id)
    
    with open(out_path, 'r') as file:
        lines = file.readlines()
    
    data = []
    
    # Process each line
    for line in lines:
        line = line.strip()
        
        # Extract sequence and SUP from the line
        sequence = line.split('#SUP:')[0].strip()
        sup = int(line.split('#SUP:')[1].strip().split('#SID:')[0].strip())
        
        # Add the data to the list
        data.append({'Sequencia': sequence, 'SUP': sup})
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    
    in_df = pd.read_csv(filepath_or_buffer=in_path, names=['column'])
    size = len(in_df)
    df['Size'] = size
    return df

def dataPreparation(df):
    df['Sequencia'] = df['Sequencia'].str.rstrip().astype(str)
    df['len'] = df['Sequencia'].str.split('-1').apply(len) - 1
    df['P'] = df['SUP'] / df['Size']
    return df

def getCombinations(items):
    combinations = []
    for i in range(len(items)-1):
        comb1 = ' '.join(items[:i+1])
        comb2 = ' '.join(items[i+1:])
        combinations.append([comb1, comb2])
        
    return combinations

def calculateLift(df, team_id):
    lift_values = []
    for _,row in tqdm(df.iterrows(), desc=" iterando sobre data frame", total=len(df)):
        s = row['Sequencia']
        p_seq = row['P']
        l = row['len']
        
        if l != 1:
            items = re.split(r'(?<=-1)\s', s)
            combinations = getCombinations(items)
       
            p_list = []
       
            for combinacao in combinations:
                c1 = combinacao[0]
                c2 = combinacao[1]
       
                p1 = df.loc[df['Sequencia'] == c1, 'P'].values[0]
                p2 = df.loc[df['Sequencia'] == c2, 'P'].values[0]
       
                p = p_seq / (p1 * p2)
                p_list.append(p)
           
            lift = np.mean(p_list)
        else:
            lift = 1
        lift_values.append(lift)
    
    df['lift'] = lift_values
    df = df[['Sequencia','SUP','lift']]
    df.to_csv("zones/lift_spmf_{}.csv".format(team_id), index=False, header=True)
    return df
    
team_ids = [1613,1673,1659,1651,1646,1631,1633,1639,1644,1623,1627,1624,1628,
            1619, 1612, 1610, 1611, 1609, 10531]
team_ids = [1625]

teams_df = {}
for team_id in tqdm(team_ids, desc="iterando pelos times", total=len(team_ids)):
    df = readData(team_id)
    df = dataPreparation(df)
    teams_df[team_id] = calculateLift(df, team_id)

