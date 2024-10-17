import os
import pandas as pd
import numpy as np
from scipy import stats


metrics = ['average_compliance', 'average_cleanliness', 'average_sanctions']


methods = ['Hybrid','Centralised-end', 'Decentralised']


data_dict = {}


results_folder = 'results' 
for folder_name in os.listdir(results_folder):
    folder_path = os.path.join(results_folder, folder_name)
    if os.path.isdir(folder_path):
        try:
            world_size, density, method = folder_name.split('_')
        except ValueError:
            continue 

        if method not in methods:
            continue 


        csv_file = os.path.join(folder_path, 'average_compliance_over_time.csv')
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            key = (world_size, density, method)
            # 将数据存入字典
            if key not in data_dict:
                data_dict[key] = []
            data_dict[key].append(df)


test_results = []


world_density_keys = set([(key[0], key[1]) for key in data_dict.keys()])

for (world_size, density) in world_density_keys:
    method_data = {}
    for method in methods:
        key = (world_size, density, method)
        if key in data_dict:
            method_dfs = data_dict[key]
            combined_df = pd.concat(method_dfs, ignore_index=True)
            avg_df = combined_df.groupby('time_step').mean().reset_index()
            method_data[method] = avg_df
    if 'Hybrid' not in method_data:
        continue 

    for metric in metrics:
        hybrid_series = method_data['Hybrid'][metric]
        for other_method in ['Centralised-end', 'Decentralised']:
            if other_method not in method_data:
                continue  
            other_series = method_data[other_method][metric]


            t_stat, p_value = stats.ttest_ind(hybrid_series, other_series, equal_var=False)


            hybrid_mean = hybrid_series.mean()
            other_mean = other_series.mean()
            mean_diff = hybrid_mean - other_mean
            pooled_sd = np.sqrt((hybrid_series.std(ddof=1) ** 2 + other_series.std(ddof=1) ** 2) / 2)
            cohen_d = mean_diff / pooled_sd


            test_results.append({
                'world_size': world_size,
                'density': density,
                'metric': metric,
                'comparison': f'Hybrid vs {other_method}',
                'hybrid_mean': round(hybrid_mean, 4),
                'other_mean': round(other_mean, 4),
                'p_value': round(p_value, 4),
                'cohen_d': round(cohen_d, 4)
            })


results_df = pd.DataFrame(test_results)


results_df = results_df[['world_size', 'density', 'metric', 'comparison', 'hybrid_mean', 'other_mean', 'p_value', 'cohen_d']]


results_df.to_csv('statistical_test_results.csv', index=False)

