## 🪵 Clasificación de Defectos en Madera con Visión por Computadora
Este proyecto implementa un sistema de clasificación de defectos en superficies de madera utilizando un modelo de visión por computadora basado en ViT (Vision Transformer). La solución incluye un pipeline de entrenamiento y una aplicación web con Streamlit para la inferencia en tiempo real.
<br><br>

## 📌 Objetivo
Desarrollar un clasificador de imágenes capaz de detectar defectos comunes en superficies de madera a partir del dataset público wood_surface_defects_split. El modelo fue entrenado utilizando la biblioteca transformers de Hugging Face con la arquitectura ViTForImageClassification.
<br><br>

## 🧰 Tecnologías utilizadas

Python 3.10<br>
PyTorch<br>
Hugging Face transformers y datasets<br>
PIL (Python Imaging Library)<br>
Streamlit para inferencia web<br>
scikit-learn para métricas<br>
MlFlow para seguimiento de experimentos y registro de modelos <br>
<br>

## 🗂️ Estructura del proyecto
```text
├── main.py                 # Script principal de entrenamiento/evaluación
├── src/
│   ├── train.py            # Funciones de entrenamiento y evaluación
│   ├── utils.py            # Funciones auxiliares
│   ├── inference.py        # Interfaz web con Streamlit para clasificación
│   ├── evaluate.py         # Funciones para evaluar en test
│   └── logger.py           # Funciones para imprimir los logs
├── outputs/
│   ├── checkpoints/        # Pesos del modelo entrenado
│   └── final_model/        # Modelo final entrenado
├── models/                 # Modelos descargado o referenciado
├── data/                   # Dataset descargado o referenciado
├── docs/                   # Documentacion (presentación, documentación, etc)
├── mlruns/                 # Carpeta de MlFlow donde almacena los resultados de los entrenamientos
└── README.md               # Este archivo
```
<br>

## 🧠 Clases del dataset
El dataset contiene las siguientes clases:

Blue_Stain<br>
Crack<br>
Dead_Knot<br>
Knot_missing<br>
Live_Knot<br>
Marrow<br>
Quartzity<br>
knot_with_crack<br>
resin<br>
normal<br>
<br>

## 📦 Requisitos
Creá y activá un entorno virtual, e instalá las dependencias:

```text 
pip install -r requirements.txt
```
<br>

## 🧠 Entrenamiento y evaluación
Para entrenar el modelo:

```text 
python main.py train
```
Esto entrenará un modelo ViT con parámetros definidos en train.py. Los checkpoints se guardarán en ./outputs/checkpoints.
<br><br>

## ✅ Testeo
Para testear el modelo:

```text 
python main.py evaluate
```
Este comando evalúa el modelo sobre el conjunto de test y reporta métricas como eval_accuracy y eval_loss.
<br><br>

## 📊 Registro de entrenamiento con MLflow
El proyecto utiliza MLflow para registrar automáticamente parámetros, métricas y modelos durante el entrenamiento. Esto permite seguir la evolución del desempeño y guardar los checkpoints de manera organizada.

Para visualizar los experimentos, ejecutar:

```text
mlflow ui
```
Abrir en el navegador:

```text
http://localhost:5000
```
Allí se pueden comparar métricas como pérdida y accuracy por época y descargar los modelos guardados.
El registro está integrado en el pipeline de entrenamiento, sin configuraciones adicionales.
<br><br>

## 🌐 Interfaz Streamlit
Para ejecutar la interfaz de clasificación de imágenes:

```text
streamlit run inference.py
```
Una vez en ejecución, se abrirá una app web en tu navegador donde podrás cargar imágenes y recibir la predicción del defecto presente.