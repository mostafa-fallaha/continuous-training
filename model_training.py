import joblib
import warnings
warnings.filterwarnings('ignore')
import data_cleaning

df = data_cleaning.data_cleaning()

# ======================== Model Learning ====================================

model = joblib.load('model/model.joblib')
metric = joblib.load('model/metric.pkl')
previous_data_len = joblib.load('model/previous_data_len.pkl')

def simulate_online_learning(df, previous_data_len):
    new_data_len = len(df)

    if new_data_len > previous_data_len:
        new_data = df.iloc[previous_data_len:]
        previous_data_len = new_data_len
        joblib.dump(previous_data_len, 'model/previous_data_len.pkl')

        for _, row in new_data.iterrows():
            x = row.drop('Daily_Avg_Installs').to_dict()
            y = row['Daily_Avg_Installs']
            y_pred = model.predict_one(x)
            model.learn_one(x, y)
            metric.update(y, y_pred)
            joblib.dump(metric, 'model/metric.pkl')

        print(f'New data processed. Current MAE: {metric.get()}')
        joblib.dump(model, 'model/model.joblib')

simulate_online_learning(df, previous_data_len)

