import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from evolvepro.src.data import load_experimental_data


def load_dataset(df, path_labels: str, 
                 threshold_hit: float)-> pd.DataFrame:
    """Loads two dataframes: one from the dms/experimental data and one from the round, 
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

#Create a dataframe in which store the results
def store_results(path: str, 
                  protein_name:str, 
                  n_rounds: int):
    """This creates a nested dictionary with the replicates and the rounds for each replicate,
    storing the df_sorted_all.csv file within it. The dict is create from the directory hiercarchy 
    in which the results are stored. The input dict must be empty. 
    Please note that the n_rounds value must be written as n_rounds+1 otherwise
    you will lose the last round
    Args: path --> directory and subdirectories where results are saved
    protein_name 
    n_rounds --> Number of rounds done with Evolve
    Return: dictionary with the same structure of the dir and sub dir"""
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
                rep_path = os.path.join(path, dir, f'{protein_name}_R{i}')
                d[dir][f'{protein_name}_R{i}']= None
                round_path = os.listdir(rep_path)
                file_path = os.path.join(rep_path, 'df_sorted_all.csv')
                d[dir][f'{protein_name}_R{i}']= pd.read_csv(file_path)
        else:
            raise RuntimeError(f'No directory ends with _rep, please rename your directory')
    return d

def create_round_file(df_labels: pd.DataFrame,
                      voi: list) -> pd.DataFrame:
    """Read and process data from Labels df and voi(variant of interest) list
    This file is used as input in EvolvePro Rounds
    Args:
    df_labels--> Dataframe from the DMS experiments
    voi-->List containing the variant of interest that the round of Evolve predicted 
    Return: A dataframe with 2 columns: one for the variant and 
    one with the activity from labels file"""
    d = {}
    for v,a in zip(df_labels['Variant'], df_labels['activity']):
        if v in voi:
            v_short = v[1:]
            d[v_short] = a
    df = pd.DataFrame.from_dict(d, orient='index')
    df.index.name = 'Variant'
    df.columns = ['activity']
    df = df.reset_index()
    return df

def read_dms_data(
    directory,
    datasets,
    model,
    experiment,
    group_columns,
    aggregate_columns,
    file_pattern="{dataset}_{model}_{experiment}.csv",
):
    """
    Read and process data from multiple CSV files.

    Args:
    directory (str): Directory containing the CSV files
    datasets (list): List of dataset names
    model (str): Model name
    experiment (str): Experiment name
    group_columns (list): Columns to group by
    aggregate_columns (list): Columns to aggregate
    file_pattern (str): File name pattern for CSV files

    Returns:
    pd.DataFrame: Processed and concatenated data from all datasets
    """
    all_dfs = []

    for dataset in datasets:
        file_name = file_pattern.format(
            dataset=dataset, model=model, experiment=experiment
        )
        file_path = os.path.join(directory, file_name)

        try:
            df = pd.read_csv(file_path)
            df = process_dataframe(df, group_columns, aggregate_columns)
            df["dataset"] = dataset
            df["model"] = model
            df["experiment"] = experiment
            all_dfs.append(df)
        except FileNotFoundError:
            print(f"File {file_name} not found. Skipping...")
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def read_exp_data(
    round_base_path, round_file_names_single, wt_fasta_path, round_file_names_multi=None
):
    """
    Read and process experimental data from multiple files.

    Args:
    round_base_path (str): Base path to the data directory containing the excel files.
    round_file_names_single (list): List of single mutant round file names.
    wt_fasta_path (str): Path to the wild-type FASTA file.
    round_file_names_multi (list): List of multi mutant round file names.

    Returns:
    pd.DataFrame: Processed and concatenated experimental data
    """

    # Load experimental data
    all_experimental_data = []
    for round_file_name in round_file_names_single:
        experimental_data = load_experimental_data(
            round_base_path, round_file_name, wt_fasta_path, single_mutant=True
        )
        all_experimental_data.append(experimental_data)

    if round_file_names_multi is not None:
        for round_file_name in round_file_names_multi:
            experimental_data = load_experimental_data(
                round_base_path, round_file_name, wt_fasta_path, single_mutant=False
            )
            all_experimental_data.append(experimental_data)

    processed_dfs = []
    # Process each round's data
    for round_num, df in enumerate(all_experimental_data, start=1):
        df_copy = df.copy()

        # Set iteration for WT in first round, exclude WT from subsequent rounds
        if round_num == 1:
            df_copy.loc[df_copy["updated_variant"] == "WT", "iteration"] = 0
        else:
            df_copy = df_copy[df_copy["updated_variant"] != "WT"]

        df_copy.loc[df_copy["updated_variant"] != "WT", "iteration"] = round_num
        df_copy["iteration"] = df_copy["iteration"].astype(float)
        df_copy.rename(columns={"updated_variant": "variant"}, inplace=True)

        processed_dfs.append(df_copy)

    # Combine all processed dataframes
    combined_df = pd.concat(processed_dfs, ignore_index=True)

    return combined_df


def process_dataframe(df, group_columns, aggregate_columns):
    """
    Process a dataframe by grouping and aggregating columns.

    Args:
    df (pd.DataFrame): Input dataframe
    group_columns (list): Columns to group by
    aggregate_columns (list): Columns to aggregate

    Returns:
    pd.DataFrame: Processed dataframe
    """

    df.replace("None", np.nan, inplace=True)
    df = df.dropna(subset=aggregate_columns, how="all")
    df[aggregate_columns] = df[aggregate_columns].apply(pd.to_numeric, errors="coerce")

    grouped = df.groupby(group_columns)
    stats = grouped[aggregate_columns].agg(["mean", "std"])
    stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]
    return stats.reset_index()


def filter_dataframe(df, conditions, output_dir=None, output_file=None):
    """
    Filter a dataframe based on conditions.

    Args:
    df (pd.DataFrame): Input dataframe
    conditions (dict): Dictionary of column-value pairs to filter on
    output_dir (str, optional): Output directory for the CSV file
    output_file (str, optional): Output file name

    Returns:
    pd.DataFrame: Filtered dataframe
    """
    for column, value in conditions.items():
        if isinstance(value, list):
            filtered_df = df[df[column].isin(value)]
        else:
            filtered_df = df[df[column] == value]

    if output_dir and output_file:
        os.makedirs(output_dir, exist_ok=True)
        filtered_df.to_csv(os.path.join(output_dir, output_file), index=False)

    return filtered_df


def save_dataframe(df, output_dir=None, output_file=None):
    """
    Save a dataframe.

    Args:
    df (pd.DataFrame): Input dataframe
    output_dir (str, optional): Output directory for the CSV file
    output_file (str, optional): Output file name

    """
    if output_dir and output_file:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, output_file), index=False)


def apply_labels(
    df, column, prefix="", suffix="", value_column=None, format_string="{}"
):
    """
    Apply labels to a DataFrame column.

    Args:
    df (pd.DataFrame): Input dataframe
    column (str): Column name to be created
    prefix (str, optional): Prefix for the label
    suffix (str, optional): Suffix for the label
    value_column (str, optional): Column to use for the label value
    format_string (str): Format string for the label value

    Returns:
    pd.DataFrame: DataFrame with the new column
    """
    if value_column is None:
        df[column] = prefix + df.index.map(lambda x: format_string.format(x)) + suffix
    else:
        df[column] = (
            prefix + df[value_column].map(lambda x: format_string.format(x)) + suffix
        )
    return df


def load_external_data(file_path, label=None, rename_columns=None):
    """
    Load external data from a CSV file, optionally add a label column, and rename columns if specified.

    Args:
    file_path (str): Path to the CSV file
    label (str, optional): Label to be added as a new column
    rename_columns (dict, optional): Dictionary of column names to rename, e.g., {'old_name': 'new_name'}

    Returns:
    pd.DataFrame: Loaded and processed dataframe
    """
    df = pd.read_csv(file_path)

    if label:
        df["label"] = label

    if rename_columns:
        df = df.rename(columns=rename_columns)

    return df


def concatenate_dataframes(dataframes, output_dir=None, output_file=None):
    """
    Concatenate a list of dataframes and optionally save the result to a CSV file.

    Args:
    dataframes (list): List of dataframes to concatenate
    output_dir (str, optional): Output directory for the CSV file
    output_file (str, optional): Output file name

    Returns:
    pd.DataFrame: Concatenated dataframe
    """
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    if output_dir and output_file:
        os.makedirs(output_dir, exist_ok=True)
        concatenated_df.to_csv(os.path.join(output_dir, output_file), index=False)

    return concatenated_df