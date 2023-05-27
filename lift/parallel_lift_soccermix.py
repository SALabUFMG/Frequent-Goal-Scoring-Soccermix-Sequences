import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import os
from joblib import Parallel, delayed
import multiprocessing

def readData(team_id):
    out_path = r"H:\Documentos\SaLab\Frequent-Goal-Scoring-Soccermix-Sequences\lift\soccermix\{}.out".format(team_id)
    in_path = r"H:\Documentos\SaLab\Frequent-Goal-Scoring-Soccermix-Sequences\lift\soccermix\{}.txt".format(team_id)
    
    with open(out_path, 'r') as file:
        lines = file.readlines()

    data = []

    # Process each line
    for line in lines:
        line = line.strip()

        # Extract sequence and SUP from the line
        sequence = line.split('#SUP:')[0].strip()
        sup = int(line.split('#SUP:')[1].strip())

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
    combinations = [[ ' '.join(items[:i+1]), ' '.join(items[i+1:]) ] for i in range(len(items)-1)]
    return combinations

def calculateLift(df, team_id):
    output_dir = '/content/soccermix'  # Absolute path to the directory
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    lift_values = np.zeros(len(df))
    df['lift'] = np.nan

    for i, row in tqdm(df.iterrows(), desc="iterando sobre data frame", total=len(df)):
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
       
                p1 = df[df['Sequencia'] == c1]['P'].values[0]
                p2 = df[df['Sequencia'] == c2]['P'].values[0]
       
                p = p_seq / (p1 * p2)
                p_list.append(p)
           
            lift = np.mean(p_list)
        else:
            lift = 1
        
        lift_values[i] = lift
    
    df['lift'] = np.where(df['len'] != 1, np.mean(p_list), 1)
    df = df[['Sequencia', 'SUP', 'lift']]
    
    csv_filename = r"H:\Documentos\SaLab\Frequent-Goal-Scoring-Soccermix-Sequences\lift\soccermix\lift_spmf_{}.csv".format(team_id)
    df.to_csv(csv_filename, index=False)

    return df

team_ids = [1613, 1673, 1659, 1651, 1646, 1631, 1633, 1639, 1644, 1623, 1625,1627, 1624, 1628, 1619, 1612, 1610, 1611, 1609, 10531]
teams_df = {}

def process_team(team_id):
    df = readData(team_id)
    df = dataPreparation(df)
    result = calculateLift(df, team_id)
    return team_id, result

# Define o número de núcleos a serem utilizados (ajuste de acordo com o seu ambiente)
num_cores = multiprocessing.cpu_count()

# Paraleliza o processamento das equipes
results = Parallel(n_jobs=num_cores)(delayed(process_team)(team_id) for team_id in tqdm(team_ids, desc="iterando pelos times", total=len(team_ids)))

# Atualiza o dicionário com os resultados
#for team_id, result in results:
    #teams_df[team_id] = result