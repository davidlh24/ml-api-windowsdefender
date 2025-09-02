import streamlit as st
import pickle
import pandas as pd
import os

# ==================================================
#  CONFIGURACIÓN DE RUTAS
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OBJ_DIR = os.path.join(BASE_DIR, "objetos")

MODEL_PATH = os.path.join(OBJ_DIR, "modelo.pkl")
FEATURES_PATH = os.path.join(OBJ_DIR, "final_features.pkl")

# ==================================================
#  CARGAR MODELO Y FEATURES
# ==================================================
with open(MODEL_PATH, "rb") as f:
    modelo = pickle.load(f)

with open(FEATURES_PATH, "rb") as f:
    final_features = pickle.load(f)

# ==================================================
#  DASHBOARD STREAMLIT
# ==================================================
st.set_page_config(page_title="Dashboard ML", layout="centered")

st.title("🔎 Dashboard de Predicciones")
st.write("Sube tus datos o ingrésalos manualmente para probar el modelo.")

# Inputs dinámicos según features
user_input = {}
for feature in final_features:
    user_input[feature] = st.number_input(f"{feature}", value=0.0)

if st.button("🔮 Predecir"):
    df = pd.DataFrame([user_input])
    pred = modelo.predict(df)[0]
    prob = modelo.predict_proba(df)[0]

    st.subheader("📊 Resultado:")
    st.write(f"**Predicción:** {pred}")
    st.write(f"**Probabilidades:** {prob}")
