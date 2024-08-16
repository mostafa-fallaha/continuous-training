import pandas as pd
import dvc.api, dvc.repo
import io
from scipy.stats import ks_2samp, chi2_contingency

# Getting the data from my storage
url = "https://github.com/mostafa-fallaha/continuous-training"
data = dvc.api.read("data/new_data.csv", encoding='utf-8', repo=url)
df_ref = pd.read_csv(io.StringIO(data))

# this is the new generated data
df_cur = pd.read_csv("data/new_data.csv")

# like a threshold
alpha = 0.05

def infer_column_type(column):
    # return column.nunique() / len(column)
    if column.dtype == 'object':
        return 'categorical'
    elif column.nunique() / len(column) < 0.001:
        return 'categorical'
    else:
        return 'continuous'

column_types = {col: infer_column_type(df_ref[col]) for col in df_ref.columns}

def chi_squared_test(ref, current):
    freq_ref = ref.value_counts().sort_index()
    freq_current = current.value_counts().sort_index()
    df_freq = pd.DataFrame({'reference': freq_ref, 'current': freq_current}).fillna(0)
    contingency_table = [df_freq['reference'].values, df_freq['current'].values]
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    return p_value

def ks_test(ref, current):
    ks_stat, p_value = ks_2samp(ref, current)
    return p_value

alpha = 0.05
results = {}
for column, col_type in column_types.items():
    if col_type == 'categorical':
        p_value = chi_squared_test(df_ref[column], df_cur[column])
    elif col_type == 'continuous':
        p_value = ks_test(df_ref[column], df_cur[column])
    results[column] = p_value < alpha

def check_drift_all_features():
    true_count = 0
    false_count = 0

    for a in results.values():
        if a:
            true_count += 1
        else:
            false_count += 1
    
    if true_count > false_count:
        return 1
    else: 
        return 0