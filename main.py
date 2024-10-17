import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from agent import CleaningAgent
from env import GridWorld
from utils import initial_map
import os

if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    random.seed(seed)

    steps = 400
    b = 0.5 #sanctioning rate[0.1,0.3,0.5,0.7,0.9]
    t = 10
    num_runs = 10
    w = 1 #observation zone [0,1,2 ...]

    methods = ["Centralised-end", "Decentralised", "Hybrid"]
    world_sizes = {
        'Small': (20, 20),
         'Large': (40, 40)
    }
    agent_densities = {
        'Sparse': {'Small': 3, 'Large': 12},
        # 'Sparse': {'Small': 5, 'Large': 50},
        'Natural': {'Small': 10, 'Large': 40},
        # 'Dense': {'Small': 20, 'Large': 500},
        'Dense': {'Small': 30, 'Large': 120}
    }

    # List to store results
    final_compliance_results = []

    # Iterate over world sizes and agent densities
    for world_size_name, (width, height) in world_sizes.items():
        for density_name, agent_nums in agent_densities.items():
            num_agents = agent_nums[world_size_name]

            # Initialize initial compliance probabilities
            initial_compliant_probs = [0.5 for _ in range(num_agents)]

            # Generate environment positions based on the world size
            house_positions, office_positions, park_positions, trash_bins = initial_map(num_agents, width, height)


            for method in methods:
                # Prepare to collect results for boxplots and time series
                compliance_over_time_runs = []
                cleanliness_over_time_runs = []
                sanctions_over_time_runs = []

                for run in range(num_runs):
                    print(f'Running {method} for run {run + 1} with {num_agents} agents in {world_size_name} world ({width}x{height}), {density_name} density')

                    env = GridWorld(
                        width, height, num_agents, method, b,
                        initial_compliant_probs, house_positions,
                        office_positions, park_positions, trash_bins,
                        run, t,w
                    )
                    env.run_env(steps)

                    # Collect final compliance probabilities
                    final_compliant_probs = [agent.compliant_prob for agent in env.agents]
                    average_compliance = sum(final_compliant_probs) / len(final_compliant_probs)

                    if env.num_agents > 0:
                        total_sanctions_received = sum(agent.sanctioned for agent in env.agents)
                        average_sanctions_per_agent = total_sanctions_received / len(env.agents)
                    else:
                        average_sanctions_per_agent = 0  # No agents left

                    # Save final average compliance for boxplots
                    final_compliance_results.append({
                        'world_size': world_size_name,
                        'density': density_name,
                        'method': method,
                        'run': run + 1,
                        'average_compliance': average_compliance,
                        'average_sanctions_per_agent': average_sanctions_per_agent,
                        # 'agents_remaining': env.num_agents,
                        # 'agents_removed': env.dead_agents_count
                    })

                    # Collect compliance over time for this run
                    compliance_over_time = env.get_average_compliance_over_time()
                    compliance_over_time_runs.append(compliance_over_time)

                    cleanliness_over_time = env.cleanliness_over_time
                    cleanliness_over_time_runs.append(cleanliness_over_time)

                    sanctions_over_time = env.total_sanctions_over_time
                    sanctions_over_time_runs.append(sanctions_over_time)

                # After all runs for this method, save the compliance over time data
                # Average over the runs
                avg_compliance_over_time = np.mean(compliance_over_time_runs, axis=0)
                avg_cleanliness_over_time = np.mean(cleanliness_over_time_runs, axis=0)
                avg_sanctions_over_time = np.mean(sanctions_over_time_runs, axis=0)


                # Save the time series data to CSV
                folder_name = f'results/{world_size_name}_{density_name}_{method}'
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                time_series_df = pd.DataFrame({
                    'time_step': np.arange(len(avg_compliance_over_time)),
                    'average_compliance': avg_compliance_over_time,
                    'average_cleanliness': avg_cleanliness_over_time,
                    'average_sanctions': avg_sanctions_over_time
                })
                time_series_df.to_csv(f'{folder_name}/3smallaverage_compliance_over_time.csv', index=False)

    # Save final compliance results to CSV for boxplots
    final_compliance_df = pd.DataFrame(final_compliance_results)
    final_compliance_df.to_csv('3smallfinal_compliance_results.csv', index=False)
