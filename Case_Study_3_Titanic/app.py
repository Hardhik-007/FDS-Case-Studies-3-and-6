"""Streamlit app: Titanic survival prediction with a logistic regression model.

Run locally:  streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "titanic_model.joblib"


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"]


def build_input(pclass, sex, age, sibsp, parch, fare, embarked, features):
    """Encode raw inputs exactly as in the notebook (female=1, Embarked one-hot with C as baseline)."""
    row = {
        "Pclass": pclass,
        "Sex": 1 if sex == "Female" else 0,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked_Q": 1 if embarked == "Queenstown (Q)" else 0,
        "Embarked_S": 1 if embarked == "Southampton (S)" else 0,
    }
    return pd.DataFrame([row])[features]


st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to estimate the probability of survival using a trained logistic regression model.")

model, features = load_model()

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Passenger class", [1, 2, 3], index=2)
    sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
    age = st.slider("Age (years)", 0.5, 80.0, 30.0, 0.5)
    embarked = st.selectbox("Port of embarkation", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"], index=2)
with col2:
    sibsp = st.number_input("Siblings / spouses aboard", min_value=0, max_value=8, value=0, step=1)
    parch = st.number_input("Parents / children aboard", min_value=0, max_value=6, value=0, step=1)
    fare = st.number_input("Ticket fare (£)", min_value=0.0, max_value=520.0, value=32.0, step=1.0)

if st.button("Predict survival", type="primary"):
    X_new = build_input(pclass, sex, age, sibsp, parch, fare, embarked, features)
    prob = float(model.predict_proba(X_new)[0, 1])
    st.metric("Estimated survival probability", f"{prob:.1%}")
    st.progress(prob)
    if prob >= 0.5:
        st.success("Prediction: this passenger would likely SURVIVE.")
    else:
        st.error("Prediction: this passenger would likely NOT survive.")
    st.caption("Classification threshold = 0.5. Predictions reflect patterns in historical data, not certainty.")
