import pandas as pd
import dvc.api, dvc.repo
import io
from pathlib import Path
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from river import linear_model, optim, metrics, preprocessing
import joblib
import warnings
warnings.filterwarnings('ignore')

url = "https://github.com/mostafa-fallaha/continuous-training"
data = dvc.api.read("data/new_data.csv", encoding='utf-8', repo=url)
df = pd.read_csv(io.StringIO(data))
print(df.shape)

# Convert boolean columns to 0 and 1
df['Free'] = df['Free'].replace({True: 1, False: 0})
df['Ad Supported'] = df['Ad Supported'].replace({True: 1, False: 0})
df['In App Purchases'] = df['In App Purchases'].replace({True: 1, False: 0})
df['Editors Choice'] = df['Editors Choice'].replace({True: 1, False: 0})

# Convert categorical variables into dummy variables
df = pd.get_dummies(df, columns=['Category', 'Content Rating'])

# Feature Scaling
scaler = StandardScaler()

# Selecting numerical columns for scaling
df.drop(columns=['Days_Between'])
numerical_columns = ['Rating', 'Rating Count', 'Installs', 'Price', 'Size_M', 'Released_Year', 'Released_Month']
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])


# X = df[df.columns.difference(['Daily_Avg_Installs'])]
# y = df.Daily_Avg_Installs

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

# ======================== Model Training ====================================

# scaler = preprocessing.StandardScaler()
# model = scaler | linear_model.LinearRegression(optimizer=optim.SGD(0.0001))
# metric = metrics.MAE()
# previous_data_len = 0

model = joblib.load('model/model.joblib')
metric = joblib.load('model/metric.pkl')
previous_data_len = joblib.load('model/previous_data_len.pkl')
print("Prev data len:", previous_data_len)
print('Prev MAE:',metric)

def simulate_online_learning(df, previous_data_len):
    new_data_len = len(df)

    if new_data_len > previous_data_len:
        new_data = df.iloc[previous_data_len:]
        previous_data_len = new_data_len
        joblib.dump(previous_data_len, 'model/previous_data_len.pkl')

        for _, row in new_data.iterrows():
            x = row.drop('Daily_Avg_Installs').to_dict()  # Replace 'Daily_Avg_Installs' with the name of your target column
            y = row['Daily_Avg_Installs']  # Replace 'Daily_Avg_Installs' with the name of your target column
            y_pred = model.predict_one(x)
            model.learn_one(x, y)
            metric.update(y, y_pred)
            joblib.dump(metric, 'model/metric.pkl')

        print(f'New data processed. Current MAE: {metric.get()}')
        joblib.dump(model, 'model/model.joblib')

simulate_online_learning(df, previous_data_len)

# # Evaluate the initial subset of data
# def initial_evaluation(df):
#     X = df[df.columns.difference(['Daily_Avg_Installs'])].head(1000)
#     y = df['Daily_Avg_Installs'].head(1000)
    
#     for x, y_true in zip(X.to_dict(orient='records'), y):
#         y_pred = model.predict_one(x)
#         model.learn_one(x, y_true)
#         metric.update(y_true, y_pred)
        
#     print(f'Initial evaluation MAE: {metric.get()}')

# # Start with an initial evaluation
# initial_evaluation(df)



# model = SGDRegressor(max_iter=1000, tol=1e-3)
# model.partial_fit(X_train, y_train)

# polynomial = PolynomialFeatures(degree=2, include_bias= False, interaction_only = False)
# X_train_poly = polynomial.fit_transform(X_train)
# X_test_poly = polynomial.transform(X_test)

# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# rmse = root_mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print("\nrmse:",rmse)
# print("r2:",r2)

# filepath = Path('data/model_data.csv')
# filepath.parent.mkdir(parents=True, exist_ok=True)
# df.to_csv(filepath, index=False)