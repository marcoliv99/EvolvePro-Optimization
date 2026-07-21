import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from evolvepro.src.data import load_experimental_data

pd.options.mode.chained_assignment = None  # default='warn'


def plot_comparison(
    concatenated_df,
    palette=None,
    variable="activity_binary_percentage_mean",
    title=None,
    output_dir=None,
    output_file=None,
):
    """
    Generate plots from a concatenated dataframe.

    Args:
    concatenated_df (pd.DataFrame): Dataframe containing the data to plot
    palette (dict, optional): Custom color palette for the plots
    variable (str): Name of the variable to plot on y-axis
    output_dir (str, optional): Directory to save the plots
    output_file (str, optional): Base name for the output files

    Returns:
    None
    """
    if palette is None:
        palette_colors = sns.color_palette("tab10")
    else:
        palette_colors = palette

    # Plot 1: Bar plot by dataset
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=concatenated_df,
        x="dataset",
        y=variable,
        hue="label",
        palette=palette_colors,
        alpha=0.75,
    )
    plt.xlabel("Dataset")
    plt.ylabel(f"{variable.replace('_', ' ').title()}")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc="upper left")

    if output_dir and output_file:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, f"{output_file}_by_dataset.png"),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

    # Plot 2: Bar plot by label with swarm plot
    plt.figure(figsize=(7, 6))
    sns.barplot(
        data=concatenated_df, x="label", y=variable, palette=palette_colors, alpha=0.75
    )
    sns.swarmplot(data=concatenated_df, x="label", y=variable, size=4, color="black")
    plt.xlabel("Model")
    plt.ylabel(f"{variable.replace('_', ' ').title()}")
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()

    if output_dir and output_file:
        plt.savefig(
            os.path.join(output_dir, f"{output_file}_by_model.png"),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def plot_grid_search_bar(
    df,
    variable="activity_binary_percentage_mean",
    strategy_column=None,
    title=None,
    output_dir=None,
    output_file=None,
):
    """
    Generate plots from a dataframe to compare different strategies.

    Args:
    df (pd.DataFrame): Dataframe containing the data to plot
    variable (str): Name of the variable to plot on y-axis
    strategy_column (str): Name of the column containing different strategies
    title (str, optional): Title for the plot
    output_dir (str, optional): Directory to save the plots
    output_file (str, optional): Base name for the output files

    Returns:
    None
    """
    if strategy_column is None:
        raise ValueError("strategy_columns must be a list of exactly two column names")

    round_num = df["round_num"].iloc[0]
    grouped = df.groupby(["dataset", strategy_column])[variable].mean().unstack()

    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x="dataset", y=variable, hue=strategy_column, alpha=0.75)

    if title is None:
        title = f"{variable.replace('_', ' ').title()} by {strategy_column.replace('_', ' ').title()} ({round_num} rounds)"

    ax.set_title(title)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(variable.replace("_", " ").title())
    ax.legend(
        title=strategy_column.replace("_", " ").title(),
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.xticks(rotation=45)
    plt.tight_layout()

    if output_dir and output_file:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, f"{output_file}_grid_bar.png"),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

    # Count the occurrences of each strategy being the best for each dataset
    winning_counts = grouped.apply(lambda x: x.idxmax(), axis=1).value_counts()

    # Print counts of winning strategies
    print("\nCounts of Winning Strategies:")
    print(winning_counts)


def plot_grid_search_heatmap(
    df,
    variable="activity_binary_percentage_mean",
    strategy_columns=None,
    title=None,
    output_dir=None,
    output_file=None,
):
    """
    Generate a heatmap from a dataframe to compare the intersection of two strategies.

    Args:
    df (pd.DataFrame): Dataframe containing the data to plot
    variable (str): Name of the variable to average and plot
    strategy_columns (list): List of two columns containing different strategies
    title (str, optional): Title for the plot
    output_dir (str, optional): Directory to save the plot
    output_file (str, optional): Base name for the output file

    Returns:
    None
    """
    if strategy_columns is None or len(strategy_columns) != 2:
        raise ValueError("strategy_columns must be a list of exactly two column names")

    # Extract round number
    round_num = df["round_num"].iloc[0]

    # Group by the two strategy columns and calculate the mean of the variable
    grouped = df.groupby(strategy_columns)[variable].mean().unstack()

    # Plot
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(grouped, cmap="viridis", annot=True, fmt=".2f", linewidths=0.5)

    if title is None:
        title = f"Average {variable.replace('_', ' ').title()} by {strategy_columns[0].replace('_', ' ').title()} and {strategy_columns[1].replace('_', ' ').title()} ({round_num} rounds)"

    ax.set_title(title)
    ax.set_xlabel(strategy_columns[1].replace("_", " ").title())
    ax.set_ylabel(strategy_columns[0].replace("_", " ").title())
    plt.tight_layout()

    if output_dir and output_file:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, f"{output_file}_grid_heatmap.png"),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def plot_by_round(
    df, variable="activity_binary_percentage_mean", output_dir=None, output_file=None
):
    """
    Plot round-by-round comparison for each dataset.

    Args:
    df (pd.DataFrame): Dataframe containing the data to plot
    variable (str): Name of the variable to plot on y-axis
    output_dir (str, optional): Directory to save the plot
    output_file (str, optional): Base name for the output file

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Generate a color palette for the unique datasets
    datasets = df["dataset"].unique()
    color_palette = sns.color_palette("tab10", n_colors=len(datasets))
    color_map = dict(zip(datasets, color_palette))

    for dataset in datasets:
        dataset_df = df[df["dataset"] == dataset]
        x_values = dataset_df["round_num"]
        y_values = dataset_df[variable]
        color = color_map[dataset]
        ax = sns.lineplot(
            x=x_values, y=y_values, ax=ax, marker="o", color=color, label=dataset
        )

    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel(variable.replace("_", " ").title())
    ax.set_title(f"{variable.replace('_', ' ').title()} by Iterations")

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if output_dir and output_file:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, f"{output_file}_by_round.png"),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def plot_by_round_split(
    df,
    variable="activity_binary_percentage_mean",
    split_variable="num_mutants_per_round",
    output_dir=None,
    output_file=None,
):
    """
    Plot round-by-round comparison with separate subplots for each dataset, with lines split by a specified variable.
    Includes a shared legend for all subplots.

    Args:
    df (pd.DataFrame): Dataframe containing the data to plot
    variable (str): Name of the variable to plot on y-axis
    split_variable (str): Name of the variable to split lines by within each subplot
    output_dir (str, optional): Directory to save the plot
    output_file (str, optional): Base name for the output file

    Returns:
    None
    """
    datasets = df["dataset"].unique()
    n_datasets = len(datasets)

    # Create color palette for split variable
    split_values = sorted(df[split_variable].unique())
    color_palette = sns.color_palette("tab10", n_colors=len(split_values))
    color_map = dict(zip(split_values, color_palette))

    # Calculate number of rows and columns for subplots
    n_cols = 3  # You can adjust this
    n_rows = (n_datasets + n_cols - 1) // n_cols

    # Create figure and subplots
    fig = plt.figure(figsize=(20, 4 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols)

    # Create plots
    for idx, dataset in enumerate(datasets):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        dataset_df = df[df["dataset"] == dataset]

        for split_value in split_values:
            subset_df = dataset_df[dataset_df[split_variable] == split_value]

            sns.lineplot(
                data=subset_df,
                x="round_num",
                y=variable,
                marker="o",
                ax=ax,
                color=color_map[split_value],
                label=f"{split_variable}: {split_value}",
            )

        ax.set_xlabel("Number of Iterations")
        ax.set_ylabel(variable.replace("_", " ").title())
        ax.set_title(dataset)
        ax.legend().remove()  # Remove individual legends

    # Remove empty subplots if any
    for idx in range(len(datasets), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(plt.subplot(gs[row, col]))

    # Create shared legend
    lines = []
    labels = []
    for split_value in split_values:
        lines.append(
            plt.Line2D(
                [0], [0], color=color_map[split_value], marker="o", linestyle="-"
            )
        )
        labels.append(f"{split_variable}: {split_value}")

    fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))

    # Add overall title
    fig.suptitle(f"{variable.replace('_', ' ').title()} by Iterations", y=1.02)

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.98, 1])

    if output_dir and output_file:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(
                output_dir, f"{output_file}_by_round_split_{split_variable}.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def plot_variants_by_iteration(
    df, df_labels:pd.DataFrame, activity_column="activity", output_dir=None, output_file=None, 
):
    """
    Simple bar plot of variants grouped by iteration.

    Args:
    df: DataFrame with 'variant', 'iteration', and activity column
    df_labels: DataFrame with activity from DMS
    activity_column: Column name containing activity values
    output_dir: Directory to save the plot
    output_file: Filename for the saved plot
    """
    # sort the dataframe by iteration and the activity column within each iteration
    df["iteration"] = df["iteration"].astype(int)
    df = df.sort_values(["iteration", activity_column], ascending=[True, True])
    df = df.reset_index(drop=True)
    activity_max_value = df_labels['activity'].max()

    plt.figure(figsize=(24, 9))

    # Plot each variant in the order of the dataframe, colored by iteration
    for iteration, group in df.groupby("iteration"):
        plt.bar(group.index, group[activity_column], label=f"Round {iteration}", width=0.6)

    # Customize
    
    plt.xticks(df.index, df["variant"], rotation=45, ha='center')
    plt.ylabel(activity_column.capitalize())
    plt.axhline(1.0, color = 'red', linestyle = '--', label='GOF cutoff')
    plt.axhline(df_labels['activity'].max(), color ='blue', linestyle = '-.', label = f'Max activity = {activity_max_value}')
    plt.margins(x = 0.005, tight=True)
    plt.legend(loc = 'lower right')

    plt.tight_layout()

    if output_dir and output_file:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, f"{output_file}_by_iteration.png"),
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()
