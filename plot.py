import matplotlib.pyplot as plt
import pandas as pd
# import wandb  # Commented out for standalone execution
import seaborn as sns

def plot_average_compliance_vs_num_agents(results_df):
    results_df['method_b'] = results_df.apply(
        lambda row: f"{row['method']} (b={row['b_value']})" if 'b_value' in row and pd.notna(row['b_value']) else row['method'],
        axis=1
    )

    plt.figure(figsize=(12, 8))
    sns.lineplot(data=results_df, x='num_agents', y='average_compliance', hue='method_b', marker='o')
    plt.xlabel('Number of Agents')
    plt.ylabel('Average Compliance Probability')
    plt.title('Average Compliance Probability vs Number of Agents')
    plt.legend(title='Method')
    plt.savefig('results/4average_compliance_vs_num_agents.png')
    plt.show()

def plot_hybrid_average_compliance_vs_b(results_df):
    hybrid_df = results_df[results_df['method'] == 'Hybrid']
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=hybrid_df, x='b_value', y='average_compliance', hue='num_agents', marker='o', palette="viridis")
    plt.xlabel('Decentralised Ratio')
    plt.ylabel('Average Compliance Probability')
    plt.title('Hybrid Method: Average Compliance Probability vs Decentralised Ratio')
    plt.legend(title='Number of Agents')
    plt.savefig('results/4Hybrid_all_agents_compliance_vs_b.png')
    plt.show()

def plot_violin_plots(compliance_df, stage_palette):
    compliance_df['method_b'] = compliance_df.apply(
        lambda row: f"{row['method']} (b={row['b_value']})" if 'b_value' in row and pd.notna(row['b_value']) else row['method'],
        axis=1
    )

    for num_agents_value in compliance_df['num_agents'].unique():
        df_subset = compliance_df[compliance_df['num_agents'] == num_agents_value]
        plt.figure(figsize=(16, 10))
        sns.violinplot(
            x='method_b',
            y='compliance_probability',
            hue='stage',
            data=df_subset,
            split=True,
            inner="quart",
            scale="width",
            palette=stage_palette
        )
        plt.title(f'Compliance Probability Distribution by Method for {num_agents_value} Agents')
        plt.xlabel('Method')
        plt.ylabel('Compliance Probability')
        plt.ylim(0, 1)
        plt.legend(title='Stage')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'results/4compliance_prob_violin_{num_agents_value}_agents.png')
        plt.show()

def plot_norm_emergence_time(results_df):
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=results_df,
        x='num_agents',
        y='norm_emergence_time',
        hue='method',
        marker='o'
    )
    plt.xlabel('Number of Agents')
    plt.ylabel('Timestep for Norm Emergence')
    plt.title('Timestep Needed for Norm Emergence')
    plt.legend(title='Method')
    plt.savefig('results/4norm_emergence_time.png')
    plt.show()

def plot_percentage_clean_cells(results_df):
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=results_df,
        x='num_agents',
        y='percentage_clean_cells',
        hue='method'
    )
    plt.xlabel('Number of Agents')
    plt.ylabel('Percentage of Clean Cells')
    plt.title('Percentage of Clean Cells After Simulation')
    plt.legend(title='Method')
    plt.savefig('results/4percentage_clean_cells.png')
    plt.show()

def plot_average_trip_duration(results_df):
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=results_df,
        x='num_agents',
        y='average_trip_duration',
        hue='method',
        marker='o'
    )
    plt.xlabel('Number of Agents')
    plt.ylabel('Average Trip Duration (Steps)')
    plt.title('Average Timestep Needed for Achieving Goals')
    plt.legend(title='Method')
    plt.savefig('results/4average_trip_duration.png')
    plt.show()


