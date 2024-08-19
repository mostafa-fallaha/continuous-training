import pandas as pd
import dvc.api, dvc.repo
import io
from sklearn.preprocessing import StandardScaler

def data_cleaning():
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
    return df