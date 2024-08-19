from river import linear_model, optim, metrics, preprocessing
import joblib
import warnings
warnings.filterwarnings('ignore')
import data_cleaning

df = data_cleaning.data_cleaning()
print(df.shape)

scaler = preprocessing.StandardScaler()
model = scaler | linear_model.LinearRegression(optimizer=optim.SGD(0.0001))
metric = metrics.MAE()
previous_data_len = 0

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