import pandas as pd
import dvc.api, dvc.repo
import io
from scipy.stats import ks_2samp

# Getting the data from my storage
url = "https://github.com/mostafa-fallaha/continuous-training"
data = dvc.api.read("data/new_data.csv", encoding='utf-8', repo=url)
df_ref = pd.read_csv(io.StringIO(data))

# this is the new generated data
df_new = pd.read_csv("data/new_data.csv")

# like a threshold
alpha = 0.05

def check_drift_all_features():
    drift_results = {}
    for feature in df_ref.columns:
        ks_stat, p_value = ks_2samp(df_ref[feature], df_new[feature])
        drift_results[feature] = p_value

    drift = 0
    no_drift = 0
    for feature, p_value in drift_results.items():
        if p_value < alpha:
            drift += 1
            # print(f"Data drift detected for feature: {feature}")
        else:
            no_drift += 1
            # print(f"No significant data drift detected for feature: {feature}")

    if drift >= no_drift:
        return 1
    else:
        return 0
