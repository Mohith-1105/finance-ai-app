import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def train_model(df):
    df['Date'] = pd.to_datetime(df['Date'])

    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['weekday'] = df['Date'].dt.weekday

    df = pd.get_dummies(df, columns=['Category'], drop_first=True)

    X = df.drop(['Amount', 'Date'], axis=1)
    y = df['Amount']

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    return model, X.columns


def predict_future_spending(model, columns, df):
    df['Date'] = pd.to_datetime(df['Date'])

    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['weekday'] = df['Date'].dt.weekday

    df = pd.get_dummies(df, columns=['Category'], drop_first=True)

    for col in columns:
        if col not in df.columns:
            df[col] = 0

    df = df[columns]

    predictions = model.predict(df)

    return np.sum(predictions)
