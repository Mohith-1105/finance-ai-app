import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import train_model, predict_future_spending
from utils import detect_anomaly, spending_risk, generate_insights

st.set_page_config(page_title="Finance AI App", layout="wide")

st.title("💰 AI Personal Finance Behavior Predictor")

# Upload CSV
uploaded_file = st.file_uploader("Upload your expense CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.subheader("Or Enter Data Manually")

    if 'data' not in st.session_state:
        st.session_state.data = pd.DataFrame(columns=['Date', 'Category', 'Amount'])

    date = st.date_input("Date")
    category = st.selectbox("Category", ["Food", "Travel", "Shopping", "Bills"])
    amount = st.number_input("Amount", min_value=0)

    if st.button("Add Expense"):
        new_row = pd.DataFrame([[date, category, amount]], columns=['Date', 'Category', 'Amount'])
        st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)

    df = st.session_state.data

if not df.empty:
    st.subheader("📊 Data Overview")
    st.dataframe(df)

    # Charts
    st.subheader("📈 Spending Trends")

    fig, ax = plt.subplots()
    df.groupby('Category')['Amount'].sum().plot(kind='bar', ax=ax)
    st.pyplot(fig)

    # Train model
    model, columns = train_model(df.copy())

    predicted = predict_future_spending(model, columns, df.copy())
    actual = df['Amount'].sum()

    st.subheader("🤖 Prediction")
    st.write(f"Predicted Future Spending: ₹{predicted:.2f}")

    risk = spending_risk(predicted, actual)
    st.write(f"⚠️ Spending Risk Level: {risk}")

    # Anomaly Detection
    st.subheader("🚨 Anomaly Detection")
    anomalies = detect_anomaly(df)

    if not anomalies.empty:
        st.write("Unusual Spending Detected:")
        st.dataframe(anomalies)
    else:
        st.write("No anomalies detected")

    # Insights
    st.subheader("💡 Insights")
    insights = generate_insights(df)
    for i in insights:
        st.write("-", i)
