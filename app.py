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

# ----------------------------
# Q5 (a) MONITORING METRICS
# ----------------------------
st.header("📈 Model Monitoring Dashboard")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Prepare simple model for monitoring
X = df[['Monthly_Charges', 'Tenure_Months']]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Metric 1: Accuracy (NOW FIXED ✔)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.metric("Model Accuracy", f"{acc:.2%}")
st.metric("F1 Score", f"{f1:.2f}")

# Metric 2: Churn Rate
churn_rate = df['Churn'].mean()
st.metric("Churn Rate", f"{churn_rate:.2%}")

# Metric 3: Data Size
st.metric("Total Customers", df.shape[0])

# Visualization
st.subheader("Churn Distribution Monitoring")
fig4, ax4 = plt.subplots()
df['Churn'].value_counts().plot(kind='bar', ax=ax4)
st.pyplot(fig4)


# ----------------------------
# Q5 (b) DATA DRIFT ANALYSIS (SIMPLE VERSION)
# ----------------------------
st.header("📉 Data Drift Analysis")

# Split dataset into "old vs new simulation"
split_point = int(len(df) * 0.7)

train_data = df.iloc[:split_point]
new_data = df.iloc[split_point:]

# Compare Monthly Charges distribution
st.subheader("Monthly Charges Distribution Shift")

fig5, ax5 = plt.subplots()

ax5.hist(train_data['Monthly_Charges'], alpha=0.5, label="Train Data")
ax5.hist(new_data['Monthly_Charges'], alpha=0.5, label="New Data")

ax5.legend()

st.pyplot(fig5)


# Simple numeric drift summary
st.write("### Summary Comparison")
st.write("Train Avg Monthly Charges:", round(train_data['Monthly_Charges'].mean(), 2))
st.write("New Avg Monthly Charges:", round(new_data['Monthly_Charges'].mean(), 2))
