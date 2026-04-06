import streamlit as st
import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.linear_model import LinearRegression

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="AI Expense Manager", layout="wide")

# -------------------- CUSTOM CSS --------------------
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .card {
        background-color: #1c1f26;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.5);
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- FIREBASE --------------------
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

# -------------------- HEADER --------------------
st.markdown("<h1 style='text-align: center;'>💰 AI Expense Manager</h1>", unsafe_allow_html=True)

# -------------------- LOAD DATA --------------------
docs = db.collection("expenses").stream()
data = [doc.to_dict() for doc in docs]

if len(data) == 0:
    st.warning("No data found in database.")
else:
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])

    # -------------------- METRICS --------------------
    total = df["amount"].sum()
    avg = df["amount"].mean()
    max_spend = df["amount"].max()

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"<div class='card'><h3>Total</h3><h2>₹ {total:.0f}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'><h3>Average</h3><h2>₹ {avg:.0f}</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'><h3>Max Spend</h3><h2>₹ {max_spend:.0f}</h2></div>", unsafe_allow_html=True)

    st.markdown("---")

    # -------------------- CHARTS --------------------
    st.markdown("---")
st.subheader("➕ Add Expense")

with st.form("expense_form"):
    date = st.date_input("Date")
    category = st.selectbox("Category", ["Food", "Travel", "Shopping", "Bills"])
    amount = st.number_input("Amount", min_value=0)

    submitted = st.form_submit_button("Add")

    if submitted:
        db.collection("expenses").add({
            "date": str(date),
            "category": category,
            "amount": float(amount)
        })
        st.success("Expense Added Successfully!")
    col1, col2 = st.columns(2)

    # Category Chart
    with col1:
        st.subheader("📌 Category Spending")
        cat = df.groupby("category")["amount"].sum()
        st.bar_chart(cat)

    # Monthly Trend
    with col2:
        st.subheader("📈 Monthly Trend")
        df["month"] = df["date"].dt.to_period("M")
        monthly = df.groupby("month")["amount"].sum()
        st.line_chart(monthly)

    st.markdown("---")

    # -------------------- DATA TABLE --------------------
    st.subheader("📊 Expense Data")
    st.dataframe(df, use_container_width=True)

    st.markdown("---")

    # -------------------- ML PREDICTION --------------------
    df = df.sort_values("date")

# Feature engineering
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek

X = df[["day", "month", "day_of_week"]]
y = df["amount"]

model = LinearRegression()
model.fit(X, y)

# Predict next 7 days
future_dates = pd.date_range(df["date"].max(), periods=8)[1:]

future_df = pd.DataFrame({
    "day": future_dates.day,
    "month": future_dates.month,
    "day_of_week": future_dates.dayofweek
})

predictions = model.predict(future_df)

st.line_chart(predictions)
st.markdown("---")
st.subheader("🚨 Unusual Spending Detection")

mean = df["amount"].mean()
std = df["amount"].std()

df["anomaly"] = df["amount"] > (mean + 2 * std)

anomalies = df[df["anomaly"] == True]

if not anomalies.empty:
    st.dataframe(anomalies)
else:
    st.write("No unusual spending detected")