import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Load dataset (IMPORTANT: must be in GitHub repo)
df = pd.read_csv("TelcoChurn_Analytics.csv")

# Basic cleaning
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

st.title("📊 Customer Churn Prediction Dashboard")

# ----------------------------
# INTERACTIVE FEATURE 1
# ----------------------------
contract = st.selectbox(
    "Select Contract Type",
    df['Contract_Type'].unique()
)

filtered = df[df['Contract_Type'] == contract]

# ----------------------------
# VISUALIZATION 1
# ----------------------------
st.subheader("1. Churn Distribution")
fig1, ax1 = plt.subplots()
df['Churn'].value_counts().plot(kind='bar', ax=ax1)
st.pyplot(fig1)

# ----------------------------
# VISUALIZATION 2
# ----------------------------
st.subheader("2. Monthly Charges Distribution")
fig2, ax2 = plt.subplots()
df['Monthly_Charges'].hist(ax=ax2)
st.pyplot(fig2)

# ----------------------------
# VISUALIZATION 3
# ----------------------------
st.subheader("3. Churn Rate by Contract Type")
fig3, ax3 = plt.subplots()
df.groupby('Contract_Type')['Churn'].mean().plot(kind='bar', ax=ax3)
st.pyplot(fig3)

# ----------------------------
# INTERACTIVE FEATURE 2
# ----------------------------
threshold = st.slider("Filter Monthly Charges", 0, 150, 50)
filtered_slider = df[df['Monthly_Charges'] > threshold]

st.write("Number of customers:", filtered_slider.shape[0])

# ----------------------------
# PREDICTION MODEL OUTPUT
# ----------------------------
st.subheader("🔮 Churn Prediction")

X = df[['Monthly_Charges', 'Tenure_Months']]
y = df['Churn']

model = LogisticRegression()
model.fit(X, y)

monthly = st.number_input("Monthly Charges", 0, 200, 50)
tenure = st.number_input("Tenure (Months)", 0, 100, 12)

if st.button("Predict"):
    pred = model.predict([[monthly, tenure]])
    result = "Will Churn ❌" if pred[0] == 1 else "Will NOT Churn ✅"
    st.success(result)
