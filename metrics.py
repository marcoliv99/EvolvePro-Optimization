import pandas as pd
import os
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
Relevance ("gain") is the the min-max-scaled true value, zeroed out below cutoff.
It is the ratio between DCG, that uses the model ranking, and IDCG, the ideal ranking
Args:
df_labels:pd.DataFrame, it is the dataframe that derives from preliminal experiments, usually DMS
df_round:pd.DataFrame, in this dataframe are present the results from the rounds
k:int, the number of top predictions to consider for the metric calculation"""
def ndcg(df_labels:pd.DataFrame, df_round: pd.DataFrame | dict, k:int=10)-> float:
    i_gain = (df_labels['activity'] - df_labels['activity'].min())/(df_labels['activity'].max()-df_labels['activity'].min())
    print(i_gain)
    IDCG = (i_gain.head(k)/np.log2(np.arange(1,k+1)+1)).sum()
    print(IDCG)
    gain =  (df_round['activity'] - df_round['activity'].min())/(df_round['activity'].max()-df_round['activity'].min())
    print(gain)
    DCG = (gain.head(k)/np.log2(np.arange(1,k+1)+1)).sum()
    print(DCG)
    NDCG = DCG/IDCG
    return NDCG

#Function that defines how to calculate the enrichment factor and the average precision
def metrics_calc(labels: str, results: dict | pd.DataFrame, rep_list: list, threshold_hit:float, output_dir: str) -> pd.DataFrame:
    """ This function takes as argument the labels file and a dictionary that stores 
    the results from the replicates of Evolvepro to calculate the Enrichment Factor
    and the Average Precision, and stores the results in a new Dataframe"""
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
            ef = enrichment_factor(df_round)
            ap = apk(df_round)
            a_rank = avg_rank(df_round)
            b_rank = best_rank(df_round)
            recall = top_recall(df_round)
            normdcg = ndcg(df_labels, df_round)
            metrics[i]['ef'].append(ef)
            metrics[i]['ap'].append(ap)
            metrics[i]['avg_rank'].append(a_rank)
            metrics[i]['best_rank'].append(b_rank)
            metrics[i]['top_recall'].append(recall)
            metrics[i]['ndcg'].append(normdcg)
    #print(df)
    #print(metrics)
    #Dict for storing the metrics results for each round and each replicate
    ef_d = {f'ef_{i}': metrics[i]['ef'] for i in rep_list}
    ap_d = {f'apk_{i}': metrics[i]['ap'] for i in rep_list}
    a_rank_d = {f'avg_rank_{i}': metrics[i]['avg_rank'] for i in rep_list}
    b_rank_d = {f'best_rank_{i}': metrics[i]['best_rank'] for i in rep_list}
    recall_d = {f'top_recall_{i}': metrics[i]['top_recall'] for i in rep_list}
    ndcg_d = {f'ndcg_{i}': metrics[i]['ndcg'] for i in rep_list}
    df_metrics = pd.DataFrame({'Rounds': rounds, 
                               **ef_d, 
                               **ap_d,
                               **a_rank_d, 
                               **b_rank_d,
                               **recall_d,
                               **ndcg_d})
    print(f'This is the df_metrics new {df_metrics}')

    
    #Calculates mean and standard deviation for each round over replicates
    df_metrics['ef_mean'] = df_metrics[list(ef_d.keys())].mean(axis=1, numeric_only=True)
    df_metrics['apk_mean'] = df_metrics[list(ap_d.keys())].mean(axis=1, numeric_only=True)
    df_metrics['avg_rank_mean'] = df_metrics[list(a_rank_d.keys())].mean(axis=1, numeric_only=True)
    df_metrics['best_rank_mean'] = df_metrics[list(b_rank_d.keys())].mean(axis=1, numeric_only=True)
    df_metrics['top_recall_mean'] = df_metrics[list(recall_d.keys())].mean(axis=1, numeric_only=True)
    df_metrics['ndcg_mean'] = df_metrics[list(ndcg_d.keys())].mean(axis=1, numeric_only=True)
    df_metrics['ef_std'] = df_metrics[list(ef_d.keys())].std(axis=1, numeric_only=True)
    df_metrics['apk_std'] = df_metrics[list(ap_d.keys())].std(axis=1, numeric_only=True)
    df_metrics['avg_rank_std'] = df_metrics[list(a_rank_d.keys())].std(axis=1, numeric_only=True)
    df_metrics['best_rank_std'] = df_metrics[list(b_rank_d.keys())].std(axis=1, numeric_only=True)
    df_metrics['top_recall_std'] = df_metrics[list(recall_d.keys())].std(axis=1, numeric_only=True)
    df_metrics['ndcg_std'] = df_metrics[list(ndcg_d.keys())].std(axis=1, numeric_only=True)
    
    df_metrics.to_csv(output_dir, index=False)
    return df_metrics


#Baseline for metrics
def random_baseline_permutation(df:pd.DataFrame, 
                                scoring_function,
                                n_permutation: int = 100,
                                random_state = None):
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


def metrics_plot(metrics_df: pd.DataFrame, 
                 path:str, 
                 mean_ef=None, 
                 mean_ap=None, 
                 rank_ef=None,
                 rank_ap=None,
                 threshold_hit=None):

    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 5))
    sns.lineplot(x=metrics_df['Rounds'], y=metrics_df['ef_mean'], marker='o', label='Mean Enrichment Factor with Std', ax=ax1)
    ax1.set_title(f'Enrichment Factor (top 10%) on rounds with threshold {threshold_hit}')
    ax1.axhline(mean_ef, color = 'black', linestyle='--', label='Baseline')
    ax1.axhline(rank_ef, color = '#009E73', linestyle = '-.', label = 'ESMRank baseline' )
    ax1.set_xlabel('Rounds')
    ax1.fill_between(x=metrics_df['Rounds'], y1=  np.subtract(metrics_df['ef_mean'], metrics_df['ef_std']), 
                     y2=np.add(metrics_df['ef_mean'], metrics_df['ef_std']),  alpha=0.2)
    ax1.set_ylabel('Enrichment Factor')
    ax1.tick_params(axis='y')
    ax1.legend()


    sns.lineplot(x=metrics_df['Rounds'], y=metrics_df['apk_mean'], marker='o', label='Mean Average Precision@10 with std', ax=ax2, color='orange')
    ax2.set_title(f'Average Precision@10 on rounds with threshold {threshold_hit}')
    ax2.set_xlabel('Rounds')
    ax2.fill_between(x=metrics_df['Rounds'], y1=  np.subtract(metrics_df['apk_mean'], metrics_df['apk_std']), 
                     y2=np.add(metrics_df['apk_mean'], metrics_df['apk_std']),  alpha=0.2)
    ax2.set_ylabel('Average Precision@10')
    ax2.tick_params(axis='y')
    ax2.axhline(mean_ap,color = 'black', linestyle='--', label='Baseline')
    ax2.axhline(rank_ap, color = "#009E73", linestyle = '-.', label = 'ESMRank baseline' )
    ax2.legend()

    fig.tight_layout()
    plt.savefig(
        path, 
        dpi=300, 
        bbox_inches='tight'
    )
    return plt.show()
