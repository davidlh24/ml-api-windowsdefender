from fastapi import FastAPI
import pickle
import pandas as pd
import os
import uvicorn

# ==================================================
#  CONFIGURACIÓN DE RUTAS
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OBJ_DIR = os.path.join(BASE_DIR, "objetos")

MODEL_PATH = os.path.join(OBJ_DIR, "modelo.pkl")
FEATURES_PATH = os.path.join(OBJ_DIR, "final_features.pkl")

# ==================================================
#  INICIALIZAR APP FASTAPI
# ==================================================
app = FastAPI(title="API del modelo ML", version="1.0")

# ==================================================
#  CARGAR MODELO Y FEATURES
# ==================================================
with open(MODEL_PATH, "rb") as f:
    modelo = pickle.load(f)

with open(FEATURES_PATH, "rb") as f:
    final_features = pickle.load(f)

# ==================================================
#  ENDPOINTS
# ==================================================
@app.get("/")
def home():
    return {"mensaje": "✅ API funcionando. Usa /predict para hacer predicciones."}

@app.post("/predict")
def predict(data: dict):
    """
    Espera un JSON con los features del modelo.
    Ejemplo de entrada:
    {
        "Feature1": 0.5,
        "Feature2": 1,
        "Feature3": "valor"
    }
    """
    # Convertir input a DataFrame
    df = pd.DataFrame([data])

    # Asegurar columnas en el mismo orden
    for col in final_features:
        if col not in df:
            df[col] = 0
    df = df[final_features]

    # Predicción
    pred = modelo.predict(df)[0]
    prob = modelo.predict_proba(df)[0].tolist()

    return {
        "prediccion": int(pred),
        "probabilidades": prob
    }

# ==================================================
#  MAIN PARA EJECUCIÓN LOCAL
# ==================================================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

