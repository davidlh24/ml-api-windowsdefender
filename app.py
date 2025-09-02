from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import os
import pandas as pd

app = FastAPI()

# Configuración CORS: React local y el mismo Render
origins = [
    #"http://localhost:5173",  React local (Vite)
    "https://davidlopez.digiservicedlh.com",
    "https://ml-api-windowsdefender.onrender.com",  # si frontend también está desplegado
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # <- aquí incluimos tus orígenes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ruta raíz
@app.get("/")
async def root():
    return {"mensaje": "✅ API funcionando. Usa /predict para hacer predicciones."}

# Endpoint de features
@app.get("/features")
async def get_features():
    obj_folder = "objetos"
    with open(os.path.join(obj_folder, "final_features.pkl"), "rb") as f:
        features = pickle.load(f)
    return features

# Endpoint de predicción
@app.post("/predict")
async def predict_endpoint(data: dict):
    obj_folder = "objetos"

    # Cargar modelo
    with open(os.path.join(obj_folder, "modelo.pkl"), "rb") as f:
        modelo = pickle.load(f)

    # Cargar features finales
    with open(os.path.join(obj_folder, "final_features.pkl"), "rb") as f:
        final_features = pickle.load(f)

    # Crear dataframe con una sola fila
    df = pd.DataFrame([data])

    # Añadir columnas faltantes con 0
    for col in final_features:
        if col not in df.columns:
            df[col] = 0

    # Reordenar columnas
    df = df[final_features]

    # Predicción
    pred = modelo.predict(df)[0]
    prob = modelo.predict_proba(df)[0]

    return {
        "prediccion": int(pred),
        "probabilidades": prob.tolist()
    }







