# mash_efficiency_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import joblib
import os

# --- Helper Functions ---
def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    df["Grain/Water Ratio"] = df["Grist Amount"] / df["Water Amount"]
    return df

def train_models(X, y):
    models = {}
    scores = {}

    for degree in [1, 2, 3]:
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        y_pred = model.predict(X_poly)
        r2 = r2_score(y, y_pred)
        models[degree] = (model, poly)
        scores[degree] = r2
    return models, scores

def predict_efficiency(models, grain_water_ratio):
    x = np.array([[grain_water_ratio]])
    preds = {}
    for degree, (model, poly) in models.items():
        x_poly = poly.transform(x)
        preds[degree] = model.predict(x_poly)[0]
    return preds

def plot_models(df, models):
    x_range = np.linspace(df["Grain/Water Ratio"].min(), df["Grain/Water Ratio"].max(), 100).reshape(-1, 1)
    plt.figure()
    plt.scatter(df["Grain/Water Ratio"], df["Mash Efficiency"], color='black', label='Data')
    for degree, (model, poly) in models.items():
        x_poly = poly.transform(x_range)
        y_pred = model.predict(x_poly)
        plt.plot(x_range, y_pred, label=f'Degree {degree}')
    plt.xlabel("Grain/Water Ratio")
    plt.ylabel("Mash Efficiency")
    plt.title("Model Comparison")
    plt.legend()
    st.pyplot(plt)

# --- Streamlit Interface ---
st.set_page_config(page_title="Mash Efficiency Predictor")
st.title("🍺 Mash Efficiency Predictor")

menu = ["Predict Efficiency", "Upload & Train", "Add Single Batch", "View Model Comparison"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Upload & Train":
    st.subheader("📤 Upload Historical Mash Data")
    uploaded_file = st.file_uploader("Choose Excel File", type=["xlsx"])
    if uploaded_file:
        df = load_data(uploaded_file)
        st.success("File uploaded and processed.")
        st.dataframe(df)

        X = df[["Grain/Water Ratio"]].values
        y = df["Mash Efficiency"].values
        models, scores = train_models(X, y)

        best_degree = max(scores, key=scores.get)
        joblib.dump(models, "models.pkl")
        joblib.dump(df, "data.pkl")

        st.success(f"Model training complete. Best degree: {best_degree} (R² = {scores[best_degree]:.3f})")

elif choice == "Add Single Batch":
    st.subheader("➕ Add a New Single Batch")
    try:
        df = joblib.load("data.pkl")
    except:
        st.warning("Please upload data first via 'Upload & Train'.")
    else:
        name = st.text_input("Batch Name")
        date = st.date_input("Date")
        grain = st.number_input("Grist Amount (kg)", min_value=0.0, step=0.1, format="%.2f")
        water = st.number_input("Water Amount (L)", min_value=0.0, step=0.1, format="%.2f")
        efficiency = st.number_input("Mash Efficiency (%)", min_value=30.0, max_value=100.0, step=0.1, format="%.1f")
        adjuncts = st.number_input("Adjuncts (% of grist)", min_value=0.0, max_value=100.0, step=0.1, format="%.1f")
        duration = st.number_input("Mash Duration (min)", min_value=0, step=1)

        if st.button("Add Batch"):
            new_row = pd.DataFrame({
                "Batch Name": [name],
                "Date": [date],
                "Grist Amount": [grain],
                "Water Amount": [water],
                "Mash Efficiency": [efficiency],
                "Adjuncts": [adjuncts],
                "Mash Duration": [duration],
                "Grain/Water Ratio": [grain / water if water > 0 else 0.0]
            })
            df = pd.concat([df, new_row], ignore_index=True)
            joblib.dump(df, "data.pkl")
            X = df[["Grain/Water Ratio"]].values
            y = df["Mash Efficiency"].values
            models, scores = train_models(X, y)
            joblib.dump(models, "models.pkl")
            st.success("New batch added and model updated.")

elif choice == "Predict Efficiency":
    st.subheader("🔮 Predict Efficiency for a New Batch")
    try:
        models = joblib.load("models.pkl")
    except:
        st.warning("Please upload and train models first.")
    else:
        grain_amount = st.number_input("Grist Amount (kg)", min_value=0.0, step=0.1, format="%.2f")
        water_amount = st.number_input("Water Amount (L)", min_value=0.0, step=0.1, format="%.2f")
        if water_amount > 0:
            ratio = grain_amount / water_amount
            preds = predict_efficiency(models, ratio)
            for degree in sorted(preds):
                efficiency_percent = preds[degree]
                st.write(f"Degree {degree} Prediction: Mash Efficiency {efficiency_percent:.1f}%")

elif choice == "View Model Comparison":
    st.subheader("📊 Model Fit Visualization")
    try:
        df = joblib.load("data.pkl")
        models = joblib.load("models.pkl")
        plot_models(df, models)
    except Exception as e:
        st.warning(f"Error loading or plotting models: {e}")
