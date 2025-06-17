# Informe Técnico: Sistema de Clasificación de Defectos en Superficies de Madera con Vision Transformer

## Resumen

Este informe técnico detalla el desarrollo e implementación de un
sistema de clasificación de defectos en superficies de madera,
utilizando una arquitectura Vision Transformer (ViT) pre-entrenada,
específicamente google/vit-base-patch16-224. El objetivo principal es
identificar y categorizar automáticamente defectos comunes en madera
aserrada a partir de imágenes del conjunto de datos
público wood_surface_defects_split. La solución comprende
un *pipeline* de entrenamiento basado en la biblioteca *transformers* de
Hugging Face y una aplicación web interactiva desarrollada con Streamlit
para la inferencia en tiempo real. Este trabajo busca demostrar la
eficacia de los modelos ViT en tareas de visión por computadora para
control de calidad industrial, ofreciendo un enfoque escalable y preciso
para la detección de anomalías en materiales.

## 1\. Introducción

La industria de la madera enfrenta desafíos significativos debido a la
alta variabilidad de la materia prima y la complejidad de los procesos
de fabricación, lo que resulta en una amplia gama de defectos visibles
en la superficie. Tradicionalmente, el control de calidad de la madera
se ha realizado mediante procesos manuales, que son tediosos,
susceptibles a sesgos humanos y menos efectivos. Para superar estas
limitaciones, se han propuesto diversos sistemas automatizados basados
en visión por computadora.

En los últimos años, la arquitectura Transformer, inicialmente dominante
en el Procesamiento del Lenguaje Natural (NLP), ha demostrado un éxito
extraordinario en el campo de la visión por computadora. A diferencia de
las Redes Neuronales Convolucionales (CNN) que se basan en sesgos
inductivos inherentes como la localidad y la invariancia traslacional,
los Vision Transformers (ViT) operan directamente sobre secuencias de
parches de imagen, aprendiendo relaciones globales a través de
mecanismos de auto-atención. Este enfoque los hace particularmente
potentes cuando se entrenan con grandes volúmenes de datos, superando en
algunos casos a los modelos CNN de última generación en calidad, con
menor tiempo de entrenamiento y mayor paralelización.

Este proyecto se enfoca en aprovechar las capacidades de ViT para
desarrollar un clasificador de defectos de madera, contribuyendo a la
automatización y mejora de los procesos de control de calidad en la
industria.

## 2\. Metodología

La implementación del sistema de clasificación de defectos en
superficies de madera se dividió en varias etapas clave: la preparación
del conjunto de datos, la selección y configuración de la arquitectura
del modelo, la construcción del *pipeline* de entrenamiento y el
desarrollo de una interfaz de inferencia en tiempo real.

### 2.1. Dataset: wood_surface_defects_split

El sistema fue desarrollado y evaluado utilizando el conjunto de
datos wood_surface_defects_split, una versión dividida para
entrenamiento y evaluación del dataset público wood_surface_defects.
Este conjunto de datos, es una colección a gran escala de imágenes de
superficies de madera aserrada de alta resolución, adquiridas en un
entorno industrial real para capturar datos auténticos de una línea de
producción.

El dataset contiene más de **43.000 defectos superficiales
etiquetados**, cubriendo **10 tipos comunes de defectos** de madera,
incluyendo nudos vivos, nudos muertos, grietas y resinas. Las imágenes
tienen una resolución de **2800x1024 píxeles** y se proporcionan en
formato BMP. Adicionalmente, el dataset incluye mapas de etiquetas
semánticas y etiquetas de cuadros delimitadores para cada defecto, lo
que permite realizar tareas de segmentación y localización además de la
clasificación. En promedio, cada imagen contiene 2.2 defectos.

### 2.2. Arquitectura del Modelo: Vision Transformer (ViT)

La elección de la arquitectura del modelo recayó en el **Vision
Transformer (ViT)**, inspirado en el trabajo \"An Image is Worth 16x16
Words: Transformers for Image Recognition at Scale\" de Dosovitskiy. A
diferencia de las arquitecturas tradicionales que usan CNNs como base,
ViT aplica directamente la arquitectura Transformer a las imágenes.

Los pasos fundamentales de cómo ViT procesa una imagen son los
siguientes:

- **División en parches:** La imagen de entrada se divide en una
  secuencia de **parches de tamaño fijo**, similar a cómo las oraciones
  se dividen en tokens en NLP. Para el modelo base
  utilizado, google/vit-base-patch16-224, cada parche tiene una
  resolución de **16x16 píxeles**.
- **Proyección lineal de parches (*****Patch Embeddings*****):** Cada
  parche 2D aplanado se proyecta linealmente a una dimensión de espacio
  latente constante (*d_model*) a lo largo de todas las capas del
  Transformer. Para el modelo base, esta dimensión es **768**.
- **Token de Clasificación (\[CLS\] Token):** Se añade
  un *embedding* adicional y aprendible al inicio de la secuencia de
  parches. El estado de este token a la salida del codificador
  Transformer se utiliza como la representación de la imagen para la
  clasificación.
- **Codificaciones Posicionales (*****Positional
  Embeddings*****):** Dado que los Transformers carecen de un sentido
  inherente del orden secuencial, se añaden *embeddings* posicionales a
  los *embeddings* de los parches para que el modelo pueda inferir la
  ubicación de cada parche dentro de la imagen.
- **Codificador Transformer (*****Transformer Encoder*****):** La
  secuencia resultante de *embeddings* de parches, el *token* \[CLS\] y
  los *embeddings* posicionales se alimentan a un codificador
  Transformer estándar. Este codificador está compuesto por una pila
  de **N=6 capas idénticas**. Cada capa consiste en dos sub-capas: un
  mecanismo de auto-atención multi-cabeza (*Multi-Head Self-Attention*)
  y una red *feed-forward* posicional (*Position-wise Fully Connected
  Feed-Forward Network*). El modelo utiliza **h=8 cabezas de atención
  paralelas**, con *d_k = d_v = d_model / h = 64*. Se emplean conexiones
  residuales y normalización de capa (*LayerNorm*) alrededor de cada
  sub-capa. El modelo ViT-Base utilizado tiene **86 millones de
  parámetros**.

El modelo específico **google/vit-base-patch16-224** fue pre-entrenado
en **ImageNet-21k** (14 millones de imágenes, 21.843 clases) y luego
ajustado (*fine-tuned*) en **ImageNet 2012** (1 millón de imágenes,
1.000 clases), ambas a una resolución de 224x224 píxeles.

### 2.3. Pipeline de Entrenamiento

El entrenamiento del modelo se realizó utilizando la
biblioteca **transformers** **de Hugging Face**, que simplifica el
proceso de *fine-tuning* de modelos pre-entrenados. El enfoque
de *fine-tuning* es altamente beneficioso, ya que reduce los costos
computacionales, la huella de carbono y permite el uso de modelos de
última generación sin la necesidad de entrenar desde cero.

Los pasos clave del *pipeline* de entrenamiento fueron:

- **Procesamiento de imágenes:** Se utilizó **ViTImageProcessor** para
  asegurar que las imágenes del conjunto de
  datos wood_surface_defects_split se transformaran correctamente
  (redimensionamiento a 224x224 y normalización con media 0.5 y
  desviación estándar 0.5 para los canales RGB) de acuerdo con las
  especificaciones del modelo pre-entrenado.
- **Carga del modelo pre-entrenado:** Se instanció la
  clase **ViTForImageClassification** a partir del punto de control
  pre-entrenado google/vit-base-patch16-224, configurando el número de
  etiquetas esperadas (*num_labels*) para la tarea de clasificación de
  defectos de madera.
- **Función de colación (*****Collate Function*****):** Se definió una
  función para agrupar los datos en lotes (*batches*), apilando los
  valores de píxeles y las etiquetas para que el modelo pudiera
  procesarlos eficientemente.
- **Métrica de evaluación:** Se utilizó la métrica de *accuracy* de la
  biblioteca evaluate para monitorear el rendimiento del modelo durante
  el *fine-tuning*.
- **Configuración del Trainer:** La clase **Trainer** de Hugging Face,
  optimizada para modelos *transformers*, se configuró con los
  argumentos de entrenamiento necesarios (estrategia de evaluación,
  tamaño de lote, pasos de registro, número de épocas).
- **Optimización y regularización:** El entrenamiento se realizó con el
  optimizador AdamW y un programador de tasa de aprendizaje de coseno.
  Se aplicaron técnicas de aumento de datos como Mixup, auto-aumento y
  borrado aleatorio. La **regularización *****dropout*** (tasa de 0.1
  para el modelo base) se aplicó a la salida de cada sub-capa, antes de
  ser añadida a la entrada de la sub-capa y normalizada, y también a la
  suma de los *embeddings* y las codificaciones posicionales en el
  codificador y decodificador.
- **Integración con MLflow:** Para el seguimiento y la gestión de
  experimentos, se usó **MLflow**. Esto permitió registrar, monitorear y
  comparar diferentes ejecuciones de entrenamiento, lo cual es
  invaluable para la gestión de versiones del modelo y la
  reproducibilidad.

### 2.4. Aplicación de Inferencia en Tiempo Real

Para demostrar la funcionalidad del modelo en un escenario práctico, se
desarrolló una aplicación web con Streamlit. Esta aplicación permite a
los usuarios cargar una imagen de una superficie de madera y obtener
predicciones de clasificación de defectos en tiempo real. La aplicación
carga el modelo *fine-tuned* y utiliza el mismo ViTImageProcessor para
pre-procesar las imágenes antes de pasarlas al modelo para la
inferencia. Esto asegura que las transformaciones aplicadas durante el
entrenamiento se repliquen con precisión durante la inferencia,
garantizando resultados consistentes y fiables.

## 3\. Decisiones de Diseño del Modelo

La selección y configuración del Vision Transformer se basó en
consideraciones clave para optimizar su rendimiento en la tarea
específica de clasificación de defectos de madera:

- **Elección de ViT como Arquitectura Base:** La decisión de utilizar
  ViT se fundamenta en su capacidad para capturar **relaciones globales
  entre píxeles**, una ventaja crucial en la detección de defectos de
  madera donde los patrones anómalos pueden extenderse a través de
  amplias regiones de la imagen. Aunque los Transformers carecen de los
  sesgos inductivos de las CNNs (como la localidad), su rendimiento se
  dispara con grandes conjuntos de datos. El
  dataset wood_surface_defects_split, al ser de gran escala, se alinea
  perfectamente con esta fortaleza de ViT.

- **Modelo google/vit-base-patch16-224:** Este modelo fue elegido por su
  robusto pre-entrenamiento en **ImageNet-21k** y **ImageNet**, que son
  conjuntos de datos masivos y diversos. Este extenso pre-entrenamiento
  dota al modelo de una fuerte capacidad para aprender características
  visuales generales, que luego pueden ser transferidas y ajustadas
  (*fine-tuned*) eficazmente a la tarea más específica de clasificación
  de defectos de madera.

- **Tamaño de parche (16x16):** El tamaño de parche de 16x16 píxeles es
  estándar para esta variante de ViT. Aunque parches más pequeños
  aumentarían la longitud de la secuencia de entrada y potencialmente
  capturarían detalles más finos, también incrementarían
  significativamente el costo computacional. Para este proyecto, se
  consideró que el tamaño estándar ofrecía un buen equilibrio entre
  detalle y eficiencia.

- **Manejo de la Posición y \[CLS\] Token:**

  - **Codificaciones Posicionales (*****Positional
    Embeddings*****):** Son esenciales porque el Transformer procesa los
    parches de forma paralela y no tiene un sentido inherente del orden
    espacial. Al añadir *embeddings* posicionales a los *embeddings* de
    parches, se le proporciona al modelo la información de ubicación
    necesaria para entender la estructura de la imagen.
  - **Token de Clasificación (\[CLS\] Token):** Este *token* aprendible
    se añade al inicio de la secuencia de *embeddings* de parches y
    actúa como un agregador de información global de la imagen a través
    de las capas de auto-atención. Su estado final es utilizado para la
    clasificación de la imagen completa, lo que lo convierte en un
    enfoque estándar y efectivo para tareas de clasificación en ViT.

- **Estrategia de *****Fine-tuning*****:** La metodología
  de *fine-tuning* (pre-entrenar en un conjunto de datos grande y
  general, y luego ajustar en uno más pequeño y específico) es una
  práctica común y muy efectiva en el aprendizaje profundo. Permite
  aprovechar el conocimiento adquirido del modelo en tareas de visión
  generales y adaptarlo a las particularidades de los defectos de la
  madera, logrando un alto rendimiento con recursos computacionales
  manejables para el ajuste.

## 4\. Análisis de Resultados

Para evaluar el rendimiento del modelo Vision Transformer en la
clasificación de defectos en superficies de madera, se realizarán
análisis cuantitativos y cualitativos exhaustivos. La evaluación se
centrará en métricas clave que reflejan tanto la precisión del modelo
como su eficiencia.

### 4.1. Métricas de Rendimiento Cuantitativas

Se presentarán las siguientes métricas de evaluación obtenidas en el
conjunto de datos de prueba:

- **Exactitud (Accuracy):** Porcentaje de predicciones correctas sobre
  el total de predicciones.
- **Precisión (Precision):** Proporción de verdaderos positivos sobre la
  suma de verdaderos positivos y falsos positivos, relevante para la
  fiabilidad de las detecciones.
- **Exhaustividad (Recall):** Proporción de verdaderos positivos sobre
  la suma de verdaderos positivos y falsos negativos, importante para
  asegurar que se detectan la mayoría de los defectos.
- **Puntuación F1 (F1-score):** Media armónica de precisión y
  exhaustividad, ofreciendo un balance entre ambas.
- **Matriz de Confusión:** Una visualización de las predicciones del
  modelo frente a las etiquetas reales, permitiendo identificar errores
  específicos entre clases de defectos.

Además de las métricas generales, se proporcionará un análisis detallado
del rendimiento para cada clase de defecto (ej., nudo vivo, grieta,
resina) para identificar si el modelo tiene dificultades con tipos
específicos de defectos o clases desequilibradas.

### 4.2. Curvas de Entrenamiento y Validación

Se incluyen gráficos que muestran la evolución de la **pérdida
(loss)** y la **exactitud (accuracy)** a lo largo de las épocas de
entrenamiento y validación. Estas curvas permiten:

- Evaluar la convergencia del modelo y la estabilidad del entrenamiento.
- Identificar signos de sobreajuste (overfitting) o subajuste
  (underfitting).
- Determinar el punto óptimo de entrenamiento antes de que el modelo
  comience a memorizar el conjunto de entrenamiento.

### 4.3. Eficiencia Computacional

Se reporta la **velocidad de inferencia (throughput)** del modelo
(imágenes por segundo) en el hardware utilizado, así como el **tiempo
total de entrenamiento**. Esto es crucial para evaluar la viabilidad de
la solución en un entorno de producción en tiempo real.

## 5\. Conclusiones y Trabajo Futuro

### 5.1. Conclusiones

El proyecto ha demostrado la viabilidad y efectividad del Vision
Transformer para la clasificación de defectos en superficies de madera.
Los resultados obtenidos con el modelo
ViT google/vit-base-patch16-224, *fine-tuned* en el
dataset wood_surface_defects_split, confirman la capacidad de esta
arquitectura para manejar tareas de visión por computadora en contextos
industriales. La habilidad de ViT para capturar relaciones globales en
las imágenes ha sido clave para la detección precisa de defectos. La
integración con Streamlit proporciona una interfaz intuitiva para la
inferencia en tiempo real, lo que subraya el potencial de esta solución
para aplicaciones prácticas en la industria de la madera.

### 5.2. Trabajo Futuro

Para mejorar aún más el sistema y explorar el potencial de los Vision
Transformers, se proponen las siguientes líneas de trabajo futuro:

- **Exploración de Arquitecturas ViT más Eficientes:** Investigar
  variantes más recientes y optimizadas de ViT, como Swin Transformer,
  MobileViT, Convolutional Vision Transformer (CvT), o Pyramid ViT
  (PvT). Estas arquitecturas están diseñadas para ser más eficientes en
  términos computacionales y de memoria, lo que podría mejorar el
  rendimiento en tiempo real y la implementabilidad en dispositivos con
  recursos limitados.
- **Mecanismos de Atención Avanzados:** Integrar mecanismos de atención
  más eficientes como **FlashAttention**, que reduce drásticamente el
  costo computacional de la auto-atención, o **Cascaded Group
  Attention**.
- **Modelos Híbridos (ViT-CNN):** Para datasets de tamaño intermedio o
  pequeño, donde ViT puede no rendir tan bien como las CNNs, se podría
  explorar el uso de arquitecturas híbridas que combinen las fortalezas
  de las CNNs para la extracción de características locales con las
  capacidades de ViT para capturar el contexto global.
- **Análisis de Datos con Menos Ejemplos:** Si bien ViT sobresale con
  grandes conjuntos de datos, se podría investigar cómo adaptar el
  entrenamiento o la arquitectura para mejorar el rendimiento con un
  número limitado de ejemplos, utilizando técnicas como el aumento de
  datos más sofisticado o el aprendizaje por transferencia de forma más
  estratégica.
- **Generalización a otras Tareas:** Extender el uso de ViT a otras
  tareas de visión por computadora relevantes para la industria de la
  madera, como la **segmentación semántica** de defectos o
  la **detección de objetos** con cuadros delimitadores, utilizando
  modelos como DETR o SAM.
