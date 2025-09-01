from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import os
import pandas as pd

app = FastAPI()

# Configuración CORS
origins = [
    "http://localhost:5173",  # React local
    "http://127.0.0.1:3000",
    "https://tu-dominio-react.netlify.app",  # si luego despliegas React
]

# origins = [
#     "http://localhost:5173",  # puerto de Vite/React
#     "https://ml-api-windowsdefender.onrender.com",  # si tu frontend también está desplegado
# ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"]
    # allow_origins=origins,
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



# from fastapi import FastAPI
# import pickle
# import pandas as pd
# import os
# import uvicorn

# # ==================================================
# #  CONFIGURACIÓN DE RUTAS
# # ==================================================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# OBJ_DIR = os.path.join(BASE_DIR, "objetos")

# MODEL_PATH = os.path.join(OBJ_DIR, "modelo.pkl")
# FEATURES_PATH = os.path.join(OBJ_DIR, "final_features.pkl")

# # ==================================================
# #  INICIALIZAR APP FASTAPI
# # ==================================================
# app = FastAPI(title="API del modelo ML", version="1.0")

# # ==================================================
# #  CARGAR MODELO Y FEATURES
# # ==================================================
# with open(MODEL_PATH, "rb") as f:
#     modelo = pickle.load(f)

# with open(FEATURES_PATH, "rb") as f:
#     final_features = pickle.load(f)

# # ==================================================
# #  ENDPOINTS
# # ==================================================
# @app.get("/")
# def home():
#     return {"mensaje": "✅ API funcionando. Usa /predict para hacer predicciones."}

# @app.post("/predict")
# def predict(data: dict):
#     """
#     Espera un JSON con los features del modelo.
#     Ejemplo de entrada:
#     {
#         "Feature1": 0.5,
#         "Feature2": 1,
#         "Feature3": "valor"
#     }
#     """
#     # Convertir input a DataFrame
#     df = pd.DataFrame([data])

#     # Asegurar columnas en el mismo orden
#     for col in final_features:
#         if col not in df:
#             df[col] = 0
#     df = df[final_features]

#     # Predicción
#     pred = modelo.predict(df)[0]
#     prob = modelo.predict_proba(df)[0].tolist()

#     return {
#         "prediccion": int(pred),
#         "probabilidades": prob
#     }

# # ==================================================
# #  MAIN PARA EJECUCIÓN LOCAL
# # ==================================================
# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)





