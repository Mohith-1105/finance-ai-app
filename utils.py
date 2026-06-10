import pandas as pd

def detect_anomaly(df):
    threshold = df['Amount'].mean() + 2 * df['Amount'].std()
    anomalies = df[df['Amount'] > threshold]
    return anomalies

def spending_risk(predicted, actual):
    if predicted > actual * 1.2:
        return "HIGH"
    elif predicted > actual:
        return "MEDIUM"
    else:
        return "LOW"

def generate_insights(df):
    insights = []

    category_spend = df.groupby('Category')['Amount'].sum()
    top_category = category_spend.idxmax()

    insights.append(f"Highest spending category: {top_category}")

    if category_spend[top_category] > df['Amount'].sum() * 0.4:
        insights.append(f"You are spending too much on {top_category}")

    return insights
