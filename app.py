import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- LOAD MODEL ----------------
with open("salary_model.pkl", "rb") as file:
    saved_data = pickle.load(file)
    model = saved_data["model"]
    scaler = saved_data["scaler"]
    label_encoders = saved_data["label_encoders"]

# Load dataset
data = pd.read_csv("adult 3.csv")

# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["📊 Data Analytics", "💼 Income Prediction"])

# ---------------- DATA ANALYTICS ----------------
if page == "📊 Data Analytics":
    st.title("📊 Employee Income Data Analytics")

    # Dataset Preview
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(data.describe())

    # Income distribution
    st.subheader("Income Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=data, x="income", ax=ax, palette="viridis")
    st.pyplot(fig)

    # Age vs Income
    st.subheader("Age vs Income")
    fig, ax = plt.subplots()
    sns.boxplot(data=data, x="income", y="age", ax=ax, palette="Set2")
    st.pyplot(fig)

    # Education vs Income
    st.subheader("Education vs Income")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=data, x="educational-num", y="age", hue="income", ax=ax)
    st.pyplot(fig)

    # Hours worked vs Income
    st.subheader("Hours per Week vs Income")
    fig, ax = plt.subplots()
    sns.boxplot(data=data, x="income", y="hours-per-week", ax=ax)
    st.pyplot(fig)

    st.success("✅ Data analytics displayed successfully!")

# ---------------- INCOME PREDICTION ----------------
elif page == "💼 Income Prediction":
    st.title("💼 Income Prediction App")
    st.write("Fill in the details to predict whether a person earns **>50K** or **<=50K**.")

    # Input fields
    age = st.slider("Age", 18, 90, 30)
    education_num = st.slider("Education Level (numeric)", 1, 16, 10)
    hours_per_week = st.slider("Work Hours per Week", 1, 100, 40)
    capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, max_value=5000, value=0)

    workclass_options = list(label_encoders["workclass"].classes_)
    occupation_options = list(label_encoders["occupation"].classes_)

    workclass = st.selectbox("Workclass", workclass_options)
    occupation = st.selectbox("Occupation", occupation_options)

    if st.button("Predict Income"):
        try:
            workclass_num = label_encoders["workclass"].transform([workclass])[0]
            occupation_num = label_encoders["occupation"].transform([occupation])[0]

            features = np.array([[age, education_num, hours_per_week, capital_gain, capital_loss, workclass_num, occupation_num]])
            features_scaled = scaler.transform(features)

            prediction = model.predict(features_scaled)[0]

            if prediction == 1:
                st.success("✅ This person is likely to earn **>50K** 💰")
            else:
                st.info("ℹ️ This person is likely to earn **<=50K**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
