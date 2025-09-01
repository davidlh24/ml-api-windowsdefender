# API de Predicción Windows Defender

Esta es una API desarrollada con **FastAPI** para servir un modelo de Machine Learning que predice resultados relacionados con Windows Defender. El backend está preparado para conectarse con un frontend (React, por ejemplo) y permite recibir datos de features para devolver predicciones y probabilidades.

---

## 📂 Estructura del proyecto


├─ app.py # Archivo principal de FastAPI
├─ requirements.txt # Dependencias de Python
├─ objetos/ # Carpeta que contiene los modelos y features
│ ├─ modelo.pkl
│ └─ final_features.pkl
├─ .gitignore
└─ README.md
---

## ⚡ Requisitos

- Python 3.9+
- FastAPI
- Uvicorn
- Pandas
- Scikit-learn (para cargar el modelo)

Instalar dependencias:

```bash
pip install -r requirements.txt
uvicorn app:app --reload
GET /
{
  "mensaje": "✅ API funcionando. Usa /predict para hacer predicciones."
}
POST /predict
{
  "engineVersion": 1.5,
  "avInstalled": 1
}
{
  "prediccion": 1,
  "probabilidades": [0.2, 0.8]
}
