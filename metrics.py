import pandas as pd
import os
import re 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from evolvepro.src.utils import load_dataset

#Enrichment Factor
def enrichment_factor(df: pd.DataFrame | dict,
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
def apk(df: pd.DataFrame | dict, 
        k: int=10) -> float:
    """
    Computes the Average Precision at k (AP@k) for a ranked list of predictions.
    AP@k measures the average precision of retrieving active compounds in the top k predictions.
    Args
    df: DataFrame sorted by predicted score in descending order, must contain a binary column 'activity_binary' (1 = active, 0 = inactive).
    k: int, the number of top predictions to consider for the AP calculation (default is 10).
    Returns
    float: The Average Precision at k (AP@k) score."""
    if df is None:
        raise RuntimeError('Error: Dataframe missing. Load a Dataframe first')
    if 'activity_binary' not in df.columns:
        raise RuntimeError('Error: The Dataframe is missing the activity_binary column.')
    r_totali = df['activity_binary'].sum()
    n_top = k
    df_top = df.head(n_top)
    somma = 0 
    hits_trovati = 0

    for i, row in enumerate(df_top.itertuples(), start=1):
        if row.activity_binary == 1:
            hits_trovati += 1 #contatore del n di hits fino alla posizione i
            precision_at_i = hits_trovati / i #calcolo della precision in pos i
            somma+= precision_at_i #contatore delle precision cumulate fino alla posizione i 
    denom = min(k, r_totali)
    apk = somma/ denom if  denom > 0 else 0.0
    return apk

#Average Ranking
"""Mean of the true rank odf the top-k prediction of the model. 
A perfect model gives (K+1)/2
Args:
df_results: pd.DataFrame, in this dataframe are present the results from the rounds
k:int,the number of top predictions to consider for the metric calculation"""
def avg_rank(df_results: pd.DataFrame | dict, k:int=10)-> float:
    avg_rank = df_results['true_rank'].head(k).mean()
    return avg_rank

#Best Ranking 
"""Best minimum true rank among the top-k predicted; a value of 1 means the model's
top-k contains the actual best variant. It's like the "did we find the winner?" metric.
Args:
df_results: pd.DataFrame, in this dataframe are present the results from the rounds
k:int,the number of top predictions to consider for the metric calculation"""
def best_rank(df:pd.DataFrame | dict, k:int=10)->int:
    best_rank = df['true_rank'].head(k).min()
    return best_rank

#Top Recall
"""This metric tells us of all the truly-good variants in the pool,
what fraction did the model place in its top-k?
Args
df_results: pd.DataFrame, in this dataframe are present the results from the rounds
k:int,the number of top predictions to consider for the metric calculation """
def top_recall(df_results: pd.DataFrame | dict, k:int=10)-> float:
    hits_top =df_results.head(k)['activity_binary'].sum()
    hits_tot = df_results['activity_binary'].sum()
    top_recall = hits_top/hits_tot
    return top_recall

#NDCG
"""This is a graded-relevance metric that rewards putting high-value variants
near the top of the predicted ranking, with a logarithimic discount for lower ranks.
Relevance ("gain") is the the min-max-scaled true value.
It is the ratio between DCG, that uses the model ranking, and IDCG, the ideal ranking
Args:
df_labels:pd.DataFrame, it is the dataframe that derives from preliminal experiments, usually DMS
df_round:pd.DataFrame, in this dataframe are present the results from the rounds
k:int, the number of top predictions to consider for the metric calculation"""
def ndcg(df_labels:pd.DataFrame, df_round: pd.DataFrame | dict, k:int=10)-> float:
    i_gain = (df_labels['activity'] - df_labels['activity'].min())/(df_labels['activity'].max()-df_labels['activity'].min())
    #print(i_gain)
    IDCG = (i_gain.head(k)/np.log2(np.arange(1,k+1)+1)).sum()
    #print(IDCG)
    gain =  (df_round['activity'] - df_round['activity'].min())/(df_round['activity'].max()-df_round['activity'].min())
    #print(gain)
    DCG = (gain.head(k)/np.log2(np.arange(1,k+1)+1)).sum()
    #print(DCG)
    NDCG = DCG/IDCG
    return NDCG

#Diversity
"""This metrics tells us how many different positions we find
in the predictions of each round
Args:
df_round:pd.DataFrame, it's the df of the single round of prediction
k:int, number of positions in the ranking that you wanna consider
round:str, which round you are passing in the function"""

def diversity(df_round: dict | pd.DataFrame, k:int):
    variant = df_round['variant']
    position_list = []
    #print(variant)
    for v in variant[:k]:
        #print(v)
        position = re.findall( r'\d+' , v)
        position_list.extend(position)
        #print(position_list)
    position_set = set(position_list)
    #print(position_set)
    #print(f'Number of  different positions in this {round}: {len(position_set)}')
    return len(position_set)


"""Auxiliary function used in the new NDCG function"""
def _minmax(x: np.ndarray) -> np.ndarray:
    """This helper function performs min-max scaling on a NumPy array, returning values in the range ([0, 1])"""
    x_min, x_max = x.min(), x.max()

    if x_max == x_min:
        return np.zeros_like(x, dtype=float)
    
    return (x - x_min) / (x_max - x_min)

#New NDCG 
"""This is a graded-relevance metric that rewards putting high-value variants
near the top of the predicted ranking, with a logarithimic discount for lower ranks.
Relevance ("gain") is the the min-max-scaled true value with the difference from the old one
that here the gains where activity is under the threshold for hits are zeroed out.
It is the ratio between DCG, that uses the model ranking, and IDCG, the ideal ranking"""
def new_ndcg(df_round: pd.DataFrame | dict, threshold_hit:float, k:int=10)-> float:
    df_ideal = df_round.copy()
    df_ideal_sorted = df_ideal.sort_values(by='activity', ascending=False).reset_index(drop=True)
    activity = np.asarray(df_ideal_sorted['activity'])
    i_gain = np.where(activity >= threshold_hit, _minmax(activity), 0.0)
    #print(f'This is the new i_gain: {i_gain}')
    IDCG = (i_gain[:k]/np.log2(np.arange(1,k+1)+1)).sum()
    #print(f'This is the new IDCG: {IDCG}')
    activity_2 = np.asarray(df_round['activity'])
    #print(activity_2)
    gain = np.where(activity_2 >= threshold_hit, _minmax(activity_2), 0.0)
    activity_2 = np.asarray(df_round['activity'])
    #print(f'This is the new gain: {gain}')
    DCG = (gain[:k]/np.log2(np.arange(1,k+1)+1)).sum()
    #print(f'This is the new DCG: {DCG}')
    NDCG = DCG/IDCG
    #print(f'This is the new NDCG: {NDCG}')
    return NDCG


"""This function calculates all the metrics for a single round of Evolvepro"""
def metrics_calc_sing(round:str , 
                      labels: str | pd.DataFrame, 
                      results: str| pd.DataFrame | dict, 
                      output_dir:str, 
                      threshold_hit:float,
                      k:int=10) -> pd.DataFrame:
    
    df_round = load_dataset(results, labels, threshold_hit)
    #print(df_round)
    ef = enrichment_factor(df_round,k)
    ap = apk(df_round, k)
    a_rank = avg_rank(df_round, k)
    b_rank = best_rank(df_round, k)
    recall = top_recall(df_round, k)
    newndcg = new_ndcg(df_round, threshold_hit, k)
    div = diversity(df_round, k)
    metrics = { f'EF@{k}': ef, 
                f'AP@{k}': ap, 
                f'Avg_Rank@{k}':a_rank, 
                f'Best_Rank@{k}':b_rank, 
                f'Top_Recall@{k}':recall, 
                f'NDCG@{k}':newndcg,
                f'Diversity@{k}':div} 
    #print(metrics)
    df_metrics = pd.DataFrame(metrics.values(), columns=[round], index= metrics.keys())
    print(f'This is the df_metrics new \n{df_metrics}')
    df_metrics.to_csv(output_dir, index=True)
    return df_metrics


#Function that defines how to calculate the enrichment factor, average precision, 
#average rank, best rank, top recall, NDCG, and diversity
def metrics_calc(labels: str, 
                 results: dict | pd.DataFrame, 
                 rep_list: list, 
                 threshold_hit:float, 
                 k: int) -> pd.DataFrame:
    """ This function takes as argument the labels file and a dictionary that stores 
    the results from the replicates of Evolvepro to calculate the Enrichment Factor, Average Precision,
    Average Rank, Best Rank, Top Recall and NDCG  and stores the results in a new Dataframe"""
    #Sorts the rounds's column in the dataframe with just the number of the iteration/round
    rounds = sorted(
        results['1_rep'].keys(),
        key=lambda x: int(''.join(filter(str.isdigit, x))) 
        if any(ch.isdigit() for ch in x) else x)
    #print(rounds)
    #creates a temporary dict in which we'll store results from the metrics calculation
    metrics = {i: {'ef': [], 
                'ap': [], 
                'avg_rank':[], 
                'best_rank':[], 
                'top_recall':[], 
                'ndcg':[]} for i in rep_list}   
    #print(metrics)
    #This nested loop iterates over the replicates and rounds results to calcule
    #the metrics for each round
    df_labels = pd.read_csv(labels)
    for round_name in rounds:
        for i in rep_list:
            rep_key = f'{i}_rep'
            df = results[rep_key][round_name]
            df_round = load_dataset(df, labels, threshold_hit)
            ef = enrichment_factor(df_round, k)
            ap = apk(df_round, k)
            a_rank = avg_rank(df_round, k)
            b_rank = best_rank(df_round, k)
            recall = top_recall(df_round, k)
            normdcg = new_ndcg(df_round,threshold_hit,k)
            divers = diversity(df_round, k)
    
            metrics[i]['ef'].append(ef)
            metrics[i]['ap'].append(ap)
            metrics[i]['avg_rank'].append(a_rank)
            metrics[i]['best_rank'].append(b_rank)
            metrics[i]['top_recall'].append(recall)
            metrics[i]['ndcg'].append(normdcg)
            metrics[i]['diversity'].append(divers)
    #print(df)
    #print(metrics)
    #Dict for storing the metrics results for each round and each replicate
    ef_d = {f'EF@{k}_{i}': metrics[i]['ef'] for i in rep_list}
    ap_d = {f'AP@{k}_{i}': metrics[i]['ap'] for i in rep_list}
    a_rank_d = {f'Avg_Rank@{k}_{i}': metrics[i]['avg_rank'] for i in rep_list}
    b_rank_d = {f'Best_Rank@{k}_{i}': metrics[i]['best_rank'] for i in rep_list}
    recall_d = {f'Top_Recall@{k}_{i}': metrics[i]['top_recall'] for i in rep_list}
    ndcg_d = {f'NDCG@{k}_{i}': metrics[i]['ndcg'] for i in rep_list}
    divers_d = {f'Diversity@{k}_{i}': metrics[i]['diversity'] for i in rep_list}

    df_metrics = pd.DataFrame({'Rounds': rounds, 
                               **ef_d, 
                               **ap_d,
                               **a_rank_d, 
                               **b_rank_d,
                               **recall_d,
                               **ndcg_d,
                               **divers_d})
    #print(f'This is the df_metrics new {df_metrics}')

    
    #Calculates mean and standard deviation for each round over replicates
    df_metrics[f'EF@{k}_Mean'] = df_metrics[list(ef_d.keys())].mean(axis=1, numeric_only=True)
    df_metrics[f'AP@{k}_Mean'] = df_metrics[list(ap_d.keys())].mean(axis=1, numeric_only=True)
    df_metrics[f'Avg_Rank@{k}_Mean'] = df_metrics[list(a_rank_d.keys())].mean(axis=1, numeric_only=True)
    df_metrics[f'Best_Rank@{k}_Mean'] = df_metrics[list(b_rank_d.keys())].mean(axis=1, numeric_only=True)
    df_metrics[f'Top_Recall@{k}_Mean'] = df_metrics[list(recall_d.keys())].mean(axis=1, numeric_only=True)
    df_metrics[f'NDCG@{k}_Mean'] = df_metrics[list(ndcg_d.keys())].mean(axis=1, numeric_only=True)
    df_metrics[f'Diversity@{k}_Mean'] = df_metrics[list(divers_d.keys())].mean(axis=1, numeric_only=True)
    df_metrics[f'EF@{k}_Std'] = df_metrics[list(ef_d.keys())].std(axis=1, numeric_only=True)
    df_metrics[f'AP@{k}_Std'] = df_metrics[list(ap_d.keys())].std(axis=1, numeric_only=True)
    df_metrics[f'Avg_Rank@{k}_Std'] = df_metrics[list(a_rank_d.keys())].std(axis=1, numeric_only=True)
    df_metrics[f'Best_Rank@{k}_Std'] = df_metrics[list(b_rank_d.keys())].std(axis=1, numeric_only=True)
    df_metrics[f'Top_Recall@{k}_Std'] = df_metrics[list(recall_d.keys())].std(axis=1, numeric_only=True)
    df_metrics[f'NDCG@{k}_Std'] = df_metrics[list(ndcg_d.keys())].std(axis=1, numeric_only=True)
    df_metrics[f'Diversity@{k}_Std'] = df_metrics[list(divers_d.keys())].std(axis=1, numeric_only=True)
    #df_metrics_t = df_metrics.transpose()
    #df_metrics.to_csv(output_dir, index=False)
    return df_metrics


#Baseline for metrics
def random_baseline_permutation(df:pd.DataFrame, 
                                scoring_function,
                                threshold_hit:float,
                                k:int,
                                n_permutation: int = 100,
                                random_state = None,
                                *args, **kwargs):
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
        df_permuted['activity_binary'] = (df_permuted['activity'] >= threshold_hit).astype(int)
        df_permuted["true_rank"] = df_permuted.index+1
        df_permuted['y_pred'] = df_permuted['activity']
        
        # Permutation of the 'activity' column
        df_permuted['y_pred'] = rng.permutation(df_permuted['y_pred'].values)
        df_permuted = df_permuted.sort_values(by='y_pred', ascending=False).reset_index(drop=True)

 # Compute the scoring function on the permuted and sorted dataframe
        if scoring_function == new_ndcg:
             score = scoring_function(df_permuted,
                                      threshold_hit=threshold_hit,
                                      k=k,
                                      *args,
                                      **kwargs)
        else:
             score = scoring_function(df_permuted,
                                      k,
                                      *args,
                                      **kwargs)
        scores.append(score)
    return np.array(scores)


