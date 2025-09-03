import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import os

# ==============================
# 🎨 ESTILOS VISUALES
# ==============================
st.markdown("""
    <style>
    /* Fondo con conic-gradient */
    .stApp {
        background: conic-gradient(
            from 20deg,
            rgba(100,100,98,0.9) 0%,
            rgba(100,100,98,0.9) 40%,
            rgba(150,130,100,0.9) 45%,
            rgba(150,130,100,0.9) 25%,
            rgba(100,100,98,0.9) 55%,
            rgba(100,100,98,0.9) 100%
        );
    }

    /* Títulos dorados */
    h1, h2, h3 {
        color: #DAA520 !important;
        font-family: 'Segoe UI', sans-serif;
        text-align: center;
    }

    /* Texto general gris claro */
    p, div, span, label {
        color: #E5E7EB !important;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Botones */
    button[kind="secondary"], button[kind="primary"] {
        background-color: #DAA520 !important;
        color: #0F172A !important;
        font-weight: bold;
        border-radius: 8px;
    }

    /* Inputs, sliders y selects */
    .stSlider, .stCheckbox, .stRadio, .stSelectbox, .stMultiSelect {
        background-color: rgba(30,41,59,0.7) !important;
        padding: 10px;
        border-radius: 8px;
    }

    /* Cuadros y tablas */
    .stDataFrame, .stTable, .stMarkdown, .stPlotlyChart, .stPyplot {
        background-color: rgba(30,41,59,0.85) !important;
        padding: 12px;
        border-radius: 12px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.6);
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# 🚀 DASHBOARD ORIGINAL
# ==============================
st.title("📊 Dashboard de Resultados del Modelo con Threshold Interactivo")

# 🔹 Cargar resultados
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OBJ_DIR = os.path.join(BASE_DIR, "objetos")
resultados = pd.read_csv(os.path.join(OBJ_DIR, "resultados_predicciones.csv"))

# 🔹 Slider para threshold
st.subheader("⚡ Ajusta el umbral de clasificación")
threshold = st.slider("Threshold para clase 1", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# 🔹 Recalcular predicciones según threshold
resultados["prediccion_thresh"] = (resultados["prob_1"] >= threshold).astype(int)

# 🔹 Vista previa
st.subheader("🔎 Vista previa de predicciones con threshold ajustado")
st.dataframe(resultados.head())

# 🔹 Distribución de clases (barra)
st.subheader("📊 Distribución de clases predichas (Barra)")
pred_dist = resultados["prediccion_thresh"].value_counts(normalize=True) * 100
fig1, ax1 = plt.subplots()
pred_dist.plot(kind="bar", ax=ax1, color=["#0088FE", "#FF8042"])
ax1.set_ylabel("Porcentaje (%)")
ax1.set_title(f"Distribución de clases predichas (%) - Threshold={threshold}")
st.pyplot(fig1)

# 🔹 Distribución de clases (pastel)
st.subheader("🥧 Distribución de clases predichas (Pastel)")
fig_pie, ax_pie = plt.subplots()
ax_pie.pie(pred_dist, labels=pred_dist.index, autopct='%1.1f%%', colors=["#0088FE", "#FF8042"], startangle=90)
ax_pie.set_title("Distribución (%) de clases predichas")
st.pyplot(fig_pie)

# 🔹 Promedio de probabilidades
st.subheader("🔎 Promedio de probabilidades")
promedios = resultados[["prob_0", "prob_1"]].mean()
col1, col2 = st.columns(2)
col1.metric("Probabilidad promedio clase 0", f"{promedios['prob_0']:.4f}")
col2.metric("Probabilidad promedio clase 1", f"{promedios['prob_1']:.4f}")

# 🔹 Histograma de probabilidades
st.subheader("📈 Histograma de probabilidades (Clase 1)")
fig2, ax2 = plt.subplots()
ax2.hist(resultados["prob_1"], bins=20, edgecolor="black", color="#0088FE")
ax2.set_xlabel("Probabilidad de clase 1")
ax2.set_ylabel("Frecuencia")
st.pyplot(fig2)

# 🔹 Gráfico de densidad KDE
st.subheader("📈 Densidad de probabilidades (Clase 1)")
fig_kde, ax_kde = plt.subplots()
sns.kdeplot(resultados["prob_1"], fill=True, color="#FF8042", alpha=0.4, ax=ax_kde)
ax_kde.set_xlabel("Probabilidad de clase 1")
ax_kde.set_ylabel("Densidad")
st.pyplot(fig_kde)

# 🔹 Boxplot
st.subheader("📊 Boxplot de probabilidades por clase predicha")
fig_box, ax_box = plt.subplots()
sns.boxplot(x="prediccion_thresh", y="prob_1", data=resultados, palette=["#0088FE", "#FF8042"], ax=ax_box)
ax_box.set_xlabel("Clase predicha")
ax_box.set_ylabel("Probabilidad de clase 1")
st.pyplot(fig_box)

# 🔹 Confianza promedio
st.subheader("✅ Confianza promedio del modelo")
confianza = resultados[["prob_0","prob_1"]].max(axis=1).mean()
st.metric(label="Confianza promedio", value=f"{confianza:.2f}")

# 🔹 Métricas
if "target" in resultados.columns:
    st.subheader("📈 Métricas de validación")
    acc = accuracy_score(resultados["target"], resultados["prediccion_thresh"])
    f1 = f1_score(resultados["target"], resultados["prediccion_thresh"])
    auc = roc_auc_score(resultados["target"], resultados["prob_1"])
    cm = confusion_matrix(resultados["target"], resultados["prediccion_thresh"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("F1 Score", f"{f1:.4f}")
    col3.metric("ROC-AUC", f"{auc:.4f}")

    st.subheader("📉 Matriz de confusión")
    st.write(cm)
else:
    st.info("ℹ️ No hay target disponible (modo inferencia). Solo se muestran predicciones y probabilidades.")



# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
# import os

# st.title("📊 Dashboard de Resultados del Modelo con Threshold Interactivo")

# # ======================
# # 🔹 Cargar resultados
# # ======================



# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# OBJ_DIR = os.path.join(BASE_DIR, "objetos")

# resultados = pd.read_csv(os.path.join(OBJ_DIR, "resultados_predicciones.csv"))

# # resultados = pd.read_csv("resultados_predicciones.csv")

# # ======================
# # 🔹 Slider para threshold
# # ======================
# st.subheader("⚡ Ajusta el umbral de clasificación")
# threshold = st.slider("Threshold para clase 1", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# # ======================
# # 🔹 Recalcular predicciones según threshold
# # ======================
# resultados["prediccion_thresh"] = (resultados["prob_1"] >= threshold).astype(int)

# # ======================
# # 🔹 Vista previa
# # ======================
# st.subheader("🔎 Vista previa de predicciones con threshold ajustado")
# st.dataframe(resultados.head())

# # ======================
# # 🔹 Distribución de clases (barra)
# # ======================
# st.subheader("📊 Distribución de clases predichas (Barra)")
# pred_dist = resultados["prediccion_thresh"].value_counts(normalize=True) * 100
# fig1, ax1 = plt.subplots()
# pred_dist.plot(kind="bar", ax=ax1, color=["skyblue", "salmon"])
# ax1.set_ylabel("Porcentaje (%)")
# ax1.set_title(f"Distribución de clases predichas (%) - Threshold={threshold}")
# st.pyplot(fig1)

# # ======================
# # 🔹 Distribución de clases (pastel)
# # ======================
# st.subheader("🥧 Distribución de clases predichas (Pastel)")
# fig_pie, ax_pie = plt.subplots()
# ax_pie.pie(pred_dist, labels=pred_dist.index, autopct='%1.1f%%', colors=["skyblue", "salmon"], startangle=90)
# ax_pie.set_title("Distribución (%) de clases predichas")
# st.pyplot(fig_pie)

# # ======================
# # 🔹 Promedio de probabilidades
# # ======================
# st.subheader("🔎 Promedio de probabilidades")
# promedios = resultados[["prob_0", "prob_1"]].mean()
# col1, col2 = st.columns(2)
# col1.metric("Probabilidad promedio clase 0", f"{promedios['prob_0']:.4f}")
# col2.metric("Probabilidad promedio clase 1", f"{promedios['prob_1']:.4f}")

# # ======================
# # 🔹 Histograma de probabilidades
# # ======================
# st.subheader("📈 Histograma de probabilidades (Clase 1)")
# fig2, ax2 = plt.subplots()
# ax2.hist(resultados["prob_1"], bins=20, edgecolor="black")
# ax2.set_xlabel("Probabilidad de clase 1")
# ax2.set_ylabel("Frecuencia")
# st.pyplot(fig2)

# # ======================
# # 🔹 Gráfico de densidad KDE
# # ======================
# st.subheader("📈 Densidad de probabilidades (Clase 1)")
# fig_kde, ax_kde = plt.subplots()
# sns.kdeplot(resultados["prob_1"], fill=True, color="purple", alpha=0.4, ax=ax_kde)
# ax_kde.set_xlabel("Probabilidad de clase 1")
# ax_kde.set_ylabel("Densidad")
# st.pyplot(fig_kde)

# # ======================
# # 🔹 Boxplot de probabilidades por clase predicha
# # ======================
# st.subheader("📊 Boxplot de probabilidades por clase predicha")
# fig_box, ax_box = plt.subplots()
# sns.boxplot(x="prediccion_thresh", y="prob_1", data=resultados, palette=["skyblue", "salmon"], ax=ax_box)
# ax_box.set_xlabel("Clase predicha")
# ax_box.set_ylabel("Probabilidad de clase 1")
# st.pyplot(fig_box)

# # ======================
# # 🔹 Confianza promedio del modelo
# # ======================
# st.subheader("✅ Confianza promedio del modelo")
# confianza = resultados[["prob_0","prob_1"]].max(axis=1).mean()
# st.metric(label="Confianza promedio", value=f"{confianza:.2f}")

# # ======================
# # 🔹 Métricas (solo si hay target)
# # ======================
# if "target" in resultados.columns:
#     st.subheader("📈 Métricas de validación")
#     acc = accuracy_score(resultados["target"], resultados["prediccion_thresh"])
#     f1 = f1_score(resultados["target"], resultados["prediccion_thresh"])
#     auc = roc_auc_score(resultados["target"], resultados["prob_1"])
#     cm = confusion_matrix(resultados["target"], resultados["prediccion_thresh"])

#     col1, col2, col3 = st.columns(3)
#     col1.metric("Accuracy", f"{acc:.4f}")
#     col2.metric("F1 Score", f"{f1:.4f}")
#     col3.metric("ROC-AUC", f"{auc:.4f}")

#     st.subheader("📉 Matriz de confusión")
#     st.write(cm)



