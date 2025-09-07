# Clasificador de Correos SPAM - HAM aplicando Algoritmo de Regresión Logística

Este proyecto presenta el desarrollo de un modelo de Regresión Logística para la clasificación de correos SPAM y HAM, con el objetivo de identificar y analizar cuáles son las características (features) más influyentes quie permiten al modelo tomar una decisión. Para la validación del modelo, se usó evaluadores como F1Score y matrices de confusión 

## Características Principales del Proyecto
* **Análisis Exploratorio de Datos (EDA):** Se realiza una investigación inicial del dataset para entender la distribución y las características de los datos.
* **Preprocesamiento de Datos:**
* **Entrenamiento del modelo:** Se construye y entrena un modelo de Regresión Logística utilizando la librería **Scikit-learn**.
* **Evaluación del Rendimiento:** El modelo se evalúa rigurosamente utilizando la **matriz de confusión** y el **F1-Score**, métricas clave para problemas de clasificación.
* **Análisis de Importancia de Features:** Se extraen y cuantifican los coeficientes del modelo para determinar la influencia porcentual de cada característica en la predicción final.

## Librerías Utilizadas: 
Este proyecto se basa en un conjunto de librerías de Python ampliamente utilizadas en el ecosistema de ciencia de datos. A continuación, se detalla el propósito de cada una:

| Librería / Módulo | Breve Descripción |
| :--- | :--- |
| **`pandas`**  | Utilizada para la manipulación y el análisis de datos. Su estructura principal, el **DataFrame**, es fundamental para cargar y limpiar el dataset. |
| **`numpy`**  | Proporciona soporte para arrays y matrices, junto con una vasta colección de funciones matemáticas para operar sobre ellos de manera eficiente. |
| **`matplotlib.pyplot`**  | Es la librería base para la creación de visualizaciones estáticas en Python, como gráficos de líneas, barras e histogramas. |
| **`seaborn`**  | Construida sobre Matplotlib, esta librería permite crear gráficos estadísticos más complejos y visualmente atractivos con menos código. |
| **`sklearn.model_selection`** | | Módulo que contiene herramientas para gestionar y dividir los datos antes de entrenar un modelo. Su función principal es asegurar que se evalúe el modelo de forma justa, probando con datos que nunca ha visto antes. 
| `train_test_split` | Función para dividir el dataset en subconjuntos aleatorios de **entrenamiento (train)** y **prueba (test)**, un paso crucial en el modelado. |
| **`sklearn.preprocessing`** | | Módulo para la limpieza y transformación de los datos para que el modelo lo entienda mejor y pueda funcionar de manera más eficiente. 
| `StandardScaler` | Herramienta para **estandarizar** los datos (media 0, varianza 1), mejorando el rendimiento de algoritmos como la Regresión Logística. |
| **`sklearn.linear_model`** | | Módulo que contiene los algoritmos de machine learning a entrenar, incluye modelos que se basan en relaciones lineales entre las variables. 
| `LogisticRegression` | Implementación del algoritmo de **Regresión Logística**, el modelo de clasificación seleccionado para este proyecto. |
| **`sklearn.metrics`** | | Módulo que contiene las herramientas necesarias para la evaluación y calificación del rendimiento del modelo. 
| `confusion_matrix` | Calcula la **matriz de confusión**, una tabla que resume el rendimiento del modelo mostrando aciertos y errores (TP, FP, FN, TN). |
| `classification_report`| Genera un informe de texto con las métricas clave de clasificación: **precisión, recall y f1-score** para cada clase. |
| `f1_score` | Calcula el **puntaje F1**, la media armónica de la precisión y el recall, una métrica robusta para evaluar clasificadores. |
| `accuracy_score` | Calcula la **exactitud (accuracy)**, que representa la proporción de predicciones correctas sobre el total de casos. |

## Estructura del Proyecto
* **`data/`**: Carpeta que contiene el conjunto de datos (`.csv`) utilizado.
* **`Machine_learning.ipynb`**: Notebook de Jupyter/Google Colab con todo el código fuente, desde la carga de datos hasta la evaluación final.
* **`requirements.txt`**: Archivo que lista todas las dependencias de Python para una fácil instalación del entorno.
* **`README.md`**: Documentación del proyecto que estás leyendo.

---


## Instalación y uso

Para replicar este proyecto en tu entorno local, sigue estos pasos:  

1.  **Clona el repositorio:**
    ```bash
    git clone https://github.com/Lenna888/Classifier_ML_Dsmatallana_Lelatorre_802.git
    cd Classifier_ML_Dsmatallana_Lelatorre_802
    ```

2.  **Crea un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecuta el Notebook:**
    Abre el archivo `Machine_learning.ipynb` en Jupyter Notebook, JupyterLab o Google Colab para ver y ejecutar el análisis.

---

## Resultados del Análisis

### Importancia de los Features

### Rendimiento del Modelo


### Conclusión





