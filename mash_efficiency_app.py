# mash_efficiency_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

# --- Helper Functions ---
def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    df["Mash Efficiency"] *= 100
    df["Grain/Water Ratio"] = df["Grist Amount"] / df["Water Amount"]
    return df

def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def train_models(X, y):
    models = {}
    scores = {}
    for degree in [1, 2]:
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        y_pred = model.predict(X_poly)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        adj_r2 = adjusted_r2(r2, len(y), X_poly.shape[1] - 1)
        models[degree] = (model, poly)
        scores[degree] = {"r2": r2, "adj_r2": adj_r2, "rmse": rmse}
    return models, scores

def predict_efficiency(models, grain_water_ratio):
    x = np.array([[grain_water_ratio]])
    preds = {}
    for degree, (model, poly) in models.items():
        x_poly = poly.transform(x)
        preds[degree] = model.predict(x_poly)[0]
    return preds

def plot_models(df, models, scores):
    x_range = np.linspace(df["Grain/Water Ratio"].min(), df["Grain/Water Ratio"].max(), 100).reshape(-1, 1)
    plt.figure()
    plt.scatter(df["Grain/Water Ratio"], df["Mash Efficiency"], color='black', label='Data')
    best_degree = max(scores, key=lambda d: scores[d]["adj_r2"])
    for degree, (model, poly) in models.items():
        x_poly = poly.transform(x_range)
        y_pred = model.predict(x_poly)
        sc = scores[degree]
        label = f'Degree {degree} (Adj. R² = {sc["adj_r2"]:.3f}, RMSE = {sc["rmse"]:.2f})'
        if degree == best_degree:
            label += " ← Best Fit"
        plt.plot(x_range, y_pred, label=label)
    plt.xlabel("Grain/Water Ratio")
    plt.ylabel("Mash Efficiency (%)")
    plt.title("Model Comparison")
    plt.legend()
    st.pyplot(plt)

    recommendation = f"Recommendation: Degree {best_degree} model currently explains the data best based on adjusted R²."
    st.markdown(f"**{recommendation}**")

# --- Streamlit Interface ---
st.set_page_config(page_title="Mash Efficiency Predictor", page_icon="🍺")
st.title("🍺 Mash Efficiency Predictor")

menu = ["Predict Efficiency", "Upload & Train", "Add Single Batch", "View Model Comparison", "View Batches"]
choice = st.sidebar.selectbox("Navigation", menu)

if os.path.exists("data.pkl"):
    df = joblib.load("data.pkl")
else:
    df = pd.DataFrame()

if choice == "Upload & Train":
    st.subheader("📤 Upload Historical Mash Data")
    uploaded_file = st.file_uploader("Choose Excel File", type=["xlsx"])
    if uploaded_file:
        new_df = load_data(uploaded_file)
        df = pd.concat([df, new_df], ignore_index=True)
        st.success("File uploaded and processed.")
        st.dataframe(df)
        X = df[["Grain/Water Ratio"]].values
        y = df["Mash Efficiency"].values
        models, scores = train_models(X, y)
        best_degree = max(scores, key=lambda d: scores[d]["adj_r2"])
        joblib.dump(models, "models.pkl")
        joblib.dump(scores, "scores.pkl")
        joblib.dump(df, "data.pkl")
        st.success(f"Model training complete. Best degree: {best_degree} (Adj. R² = {scores[best_degree]['adj_r2']:.3f})")

elif choice == "Add Single Batch":
    st.subheader("➕ Add a New Single Batch")
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
        joblib.dump(scores, "scores.pkl")
        st.success("New batch added and model updated.")

elif choice == "Predict Efficiency":
    st.subheader("🔮 Predict Efficiency for a New Batch")
    try:
        models = joblib.load("models.pkl")
        scores = joblib.load("scores.pkl")
    except:
        st.warning("Please upload and train models first.")
    else:
        grain_amount = st.number_input("Grist Amount (kg)", min_value=0.0, step=0.1, format="%.2f")
        water_amount = st.number_input("Water Amount (L)", min_value=0.0, step=0.1, format="%.2f")
        if water_amount > 0:
            ratio = grain_amount / water_amount
            preds = predict_efficiency(models, ratio)
            best_degree = max(scores, key=lambda d: scores[d]["adj_r2"])
            for degree in sorted(preds):
                eff = preds[degree]
                note = " ← recommended" if degree == best_degree else ""
                st.write(f"Degree {degree} Prediction: Mash Efficiency {eff:.1f}%{note}")

elif choice == "View Model Comparison":
    st.subheader("📊 Model Fit Visualization")
    try:
        models = joblib.load("models.pkl")
        scores = joblib.load("scores.pkl")
        plot_models(df, models, scores)
    except Exception as e:
        st.warning(f"Error loading or plotting models: {e}")

elif choice == "View Batches":
    st.subheader("📋 Stored Batch Data")
    if df.empty:
        st.info("No batch data available yet.")
    else:
        edit_idx = st.number_input("Select batch index to edit/delete", min_value=0, max_value=len(df)-1, step=1)
        st.dataframe(df)

        with st.expander("✏️ Edit Selected Batch"):
            name = st.text_input("Batch Name", value=df.loc[edit_idx, "Batch Name"])
            date = st.date_input("Date", value=pd.to_datetime(df.loc[edit_idx, "Date"]))
            grain = st.number_input("Grist Amount (kg)", value=float(df.loc[edit_idx, "Grist Amount"]))
            water = st.number_input("Water Amount (L)", value=float(df.loc[edit_idx, "Water Amount"]))
            efficiency = st.number_input("Mash Efficiency (%)", value=float(df.loc[edit_idx, "Mash Efficiency"]))
            adjuncts = st.number_input("Adjuncts (% of grist)", value=float(df.loc[edit_idx, "Adjuncts"]))
            duration = st.number_input("Mash Duration (min)", value=int(df.loc[edit_idx, "Mash Duration"]))
            if st.button("Update Batch"):
                df.loc[edit_idx] = [name, date, grain, water, efficiency, adjuncts, duration, grain / water if water > 0 else 0.0]
                joblib.dump(df, "data.pkl")
                st.success("Batch updated.")
            if st.button("Delete Batch"):
                df = df.drop(index=edit_idx).reset_index(drop=True)
                joblib.dump(df, "data.pkl")
                st.success("Batch deleted.")
