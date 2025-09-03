<p align="center">
  <img src="https://github.com/davidlh24/RegistroUsuario-PHP/blob/main/RegistroCI.jpg?raw=true" width="600" alt="Registro Usuario PHP" style="border-radius: 10px; background-color: #000000; padding: 10px;" />
</p>


🛡️ API de Predicción Windows Defender

🚀 Descripción  
Esta es una API desarrollada con **FastAPI** que sirve un modelo de **Machine Learning** para predecir resultados relacionados con **Windows Defender**.  
El backend está preparado para conectarse con un frontend (por ejemplo, en React) y permite enviar datos de entrada para obtener predicciones y probabilidades.

---

🛠️ Tecnologías Utilizadas  
- **Python 3.9+**  
- **FastAPI** → Framework para la creación de APIs.  
- **Uvicorn** → Servidor ASGI para ejecutar la API.  
- **Pandas** → Procesamiento de datos.  
- **Scikit-learn** → Para cargar y utilizar el modelo entrenado.  

---

💻 Características  
- Endpoint raíz `/` que confirma el funcionamiento de la API.  
- Endpoint `/predict` que recibe features y devuelve:  
  - Predicción (0 o 1).  
  - Probabilidades de cada clase.  
- Backend listo para integrarse con un frontend en tiempo real.  

---

📂 Estructura de Archivos  

