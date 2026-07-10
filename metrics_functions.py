import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

#Loading the dataset
def load_dataset(df, path_labels: str, 
                 threshold_hit: float)-> pd.DataFrame:
    """Loads the two dataframes: one from the dms/experimental data and one from the round, 
    merging the column activity from the first one into the second and calculating
    a new column that contains the activity in binary form based on a threshold. 
    Please note that the path argument can be both a str or a directed pd.DataFrame """
    
    if path_labels.endswith(('.xlsx', '.xls')):
        raise RuntimeError(f'The uploaded Dataframe is an excel file, please use a csv format')
    if df is str:
        df_round = pd.read_csv(df)
        if df.endswith(('.xlsx', '.xls')):
            raise RuntimeError(f'The uploaded Dataframe is an excel file, please use a csv format')
    else:
        df is pd.DataFrame
    df_round = df
    df_labels = pd.read_csv(path_labels)
    labels_activity = df_labels[['variant', 'activity']]
    df_merged = pd.merge(df_round, labels_activity, on='variant', how='left')
    df_ordinato = df_merged.sort_values(by='y_pred', ascending=False).reset_index(drop=True)
    df_ordinato['activity_binary'] = (df_ordinato['activity'] >= threshold_hit).astype(int)
    #print(df_ordinato.head(10))
    
    return df_ordinato


#Enrichment Factor
def enrichment_factor(df: pd.DataFrame,
                     fraction: float=0.1)-> float:
    """
    Computes the Enrichment Factor (EF) at a given fraction of the dataset.
    Parameters
    df: DataFrame sorted by predicted score in descending order.
        Must contain a binary column 'activity_binary' (1 = active, 0 = inactive).
    fraction : float, optional    
        Fraction of the dataset to consider as the top subset (default: 0.1 = top 10%).
    Returns
    float
    Enrichment Fraction for the specific fraction of the dataset.
"""
    if df is None:
        raise RuntimeError('Error: Dataframe missing. Load a Dataframe first')
    if 'activity_binary' not in df.columns:
        raise RuntimeError('Error: The Dataframe is missing the activity_binary column.') 
    n_totale=len(df)
    n_attivi=df['activity_binary'].sum()
    n_top= int(n_totale*fraction)
    hits_top =df.head(n_top)['activity_binary'].sum()
    if hits_top == 0 or n_top == 0:
        return 0.0
    ef= (hits_top/n_top) / (n_attivi/n_totale)
    return ef 

#Average Precision@k 
def apk(df: pd.DataFrame, 
        k: int=10) -> float:
    """
    Computes the Average Precision at k (AP@k) for a ranked list of predictions.
    AP@k measures the average precision of retrieving active compounds in the top k predictions.
    Parameters
    df: DataFrame sorted by predicted score in descending order, must contain a binary column 'activity_binary' (1 = active, 0 = inactive).
    k: int, the number of top predictions to consider for the AP calculation (default is 10).
    Returns
    float: The Average Precision at k (AP@k) score."""
    if df is None:
        raise RuntimeError('Error: Dataframe missing. Load a Dataframe first')
    if 'activity_binary' not in df.columns:
        raise RuntimeError('Error: The Dataframe is missing the activity_binary column.')
    
    r_totali =df['activity_binary'].sum()
    n_top = k
    df_top = df.head(n_top)

    somma = 0 
    hits_trovati = 0

    for i, row in enumerate(df_top.itertuples(), start=1):
        if row.activity_binary == 1:
            hits_trovati += 1 #contatore del n di hits fino alla posizione i
            precision_at_i = hits_trovati / i #calcolo della precision in pos i
            somma+= precision_at_i #contatore delle precision cumulate fino alla posizione i 

    apk = somma/ r_totali if r_totali > 0 else 0.0
    return apk


#Create a dataframe in which store the results
def store_results(path: str, n_rounds: int):
    """This creates a nested dictionary with the replicates and the rounds for each replicate,
    storing the df_sorted_all.csv file within it. The dict is create from the directory hiercarchy 
    in which the results are stored. The input dict must be empty. 
    Please note that the n_rounds value must be written as n_rounds+1 otherwise
    you will lose the last round"""
    d = {}
    #iterate over dir in path given in input 
    for dir in os.listdir(path):
        #for iterating over the dir
        if dir.endswith('_rep'): 
            #empty dict for storing
            d[dir] = {} 
            for i in range(1, n_rounds): #loop for iterate over rounds
                if path.endswith(('.xlsx', '.xls')):
                    raise RuntimeError(f'The uploaded Dataframe is an excel file, please use a csv format')
                rep_path = os.path.join(path, dir, f'Round{i}')
                d[dir][f'Round{i}']= None
                round_path = os.listdir(rep_path)
                file_path = os.path.join(rep_path, 'df_sorted_all.csv')
                d[dir][f'Round{i}']= pd.read_csv(file_path)
        else:
            raise RuntimeError(f'No directory ends with _rep, please rename your directory')
    return d

#Function that defines how to calculate the enrichment factor and the average precision
def metrics_calc(labels: str, results: dict) -> pd.DataFrame:
    """ This function takes as argument the labels file and a dictionary that stores 
    the results from 3 replicates of Evolvepro to calculate the Enrichment Factor
    and the Average Precision, and stores the results in a new Dataframe"""
    rep_list = [1, 2, 3]
    #Sorts the rounds's column in the dataframe with just the number of the iteration/round
    rounds = sorted(
        results['1_rep'].keys(),
        key=lambda x: int(''.join(filter(str.isdigit, x))) 
        if any(ch.isdigit() for ch in x) else x)
    print(rounds)
    #creates a temporary dict in which we'll store results from the metrics calculation
    metrics = {i: {'ef': [], 'apk': []} for i in rep_list}
    print(metrics)
    #This nested loop iterates over the replicates and rounds results to calcule
    #the metrics for each round
    for round_name in rounds:
        for i in rep_list:
            rep_key = f'{i}_rep'
            df = results[rep_key][round_name]
            df_round = load_dataset(df, labels)
            ef = enrichment_factor(df_round)
            ap = apk(df_round)
            metrics[i]['ef'].append(ef)
            metrics[i]['apk'].append(ap)
    print(df)
    print(metrics)
    #Dict for storing the metrics results for each round and each replicate
    df_metrics = pd.DataFrame({
        'Rounds': rounds,
        'ef_1': metrics[1]['ef'], 'apk_1': metrics[1]['apk'],
        'ef_2': metrics[2]['ef'],'apk_2': metrics[2]['apk'],
        'ef_3': metrics[3]['ef'],'apk_3': metrics[3]['apk'],
    })
    #Calculates mean and standard deviation for each round over replicates
    df_metrics['ef_mean'] = df_metrics[['ef_1', 'ef_2', 'ef_3']].mean(axis=1, numeric_only=True)
    df_metrics['apk_mean'] = df_metrics[['apk_1', 'apk_2', 'apk_3']].mean(axis=1, numeric_only=True)
    df_metrics['ef_std'] = df_metrics[['ef_1', 'ef_2', 'ef_3']].std(axis=1, numeric_only=True)
    df_metrics['apk_std'] = df_metrics[['apk_1', 'apk_2', 'apk_3']].std(axis=1, numeric_only=True)
    
    return df_metrics



#Baseline for metrics
def random_baseline_permutation(df:pd.DataFrame, 
                                scoring_function,
                                n_permutation: int = 100,
                                random_state: int = None):
    """
    Calculate random baseline via label permutation: shuffle activity n_permutation times,
    Parameters:
    df: DataFrame containing the data
    scoring_function: function to compute the desired metric (e.g., enrichment factor, average precision)
    n_permutation: number of random permutations to perform
    random_state: seed for random number generator
    """
    rng = np.random.default_rng(random_state)
    scores = []

    for _ in range(n_permutation):
        #Creates a copy of your dataframe to avoid modifying the original one
        df_permuted = df.copy()
        
        # Permutation of the 'activity' column
        df_permuted['activity'] = rng.permutation(df_permuted['activity'].values)

        #Organize the dataframe by activity values in descending order
        df_permuted_sorted = df_permuted.sort_values(by='activity', ascending=False).reset_index(drop=True)
        
        # Compute the scoring function on the permuted and sorted dataframe
        score = scoring_function(df_permuted_sorted)
        scores.append(score)

    return np.array(scores)


def metrics_plot(metrics_df: pd.DataFrame, path:str):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 5))
    sns.lineplot(x=metrics_df['Rounds'], y=metrics_df['ef_mean'], marker='o', label='Mean Enrichment Factor with Std', ax=ax1)
    ax1.set_title('Enrichment Factor (top 10%) Across Rounds')
    ax1.get_xticks
    ax1.set_xlabel('Rounds')
    ax1.fill_between(x=metrics_df['Rounds'], y1=  np.subtract(metrics_df['ef_mean'], metrics_df['ef_std']), 
                     y2=np.add(metrics_df['ef_mean'], metrics_df['ef_std']),  alpha=0.2)
    ax1.set_ylabel('Enrichment Factor', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')


    sns.lineplot(x=metrics_df['Rounds'], y=metrics_df['apk_mean'], marker='o', label='Mean Average Precision@10 with std', ax=ax2, color='orange')
    ax2.set_title('Average Precision@10 Across Rounds')
    ax2.set_xlabel('Rounds')
    ax2.fill_between(x=metrics_df['Rounds'], y1=  np.subtract(metrics_df['apk_mean'], metrics_df['apk_std']), 
                     y2=np.add(metrics_df['apk_mean'], metrics_df['apk_std']),  alpha=0.2)
    ax2.set_ylabel('Average Precision@10', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    fig.tight_layout()
    plt.savefig(
        path, 
        dpi=300, 
        bbox_inches='tight'
    )
    return plt.show()

def create_round_file(df_labels: pd.DataFrame, voi: list) -> pd.DataFrame:
    """This function takes as input the labels file (DMS results usually)
    and a list of variant of interest (voi) so it could generate a new file 
    with the voi in one coloumn and their activity in another"""
    d = {}
    for v,a in zip(df_labels['Variant'], df_labels['activity']):
        if v in voi:
            v_short = v[1:]
            d[v_short] = [a]
    df = pd.DataFrame.from_dict(d, orient='index', columns=['activity'])
    df.index.name = 'Variant'
    df = df.reset_index()
    df.columns = ['Variant', 'activity']
    return df
