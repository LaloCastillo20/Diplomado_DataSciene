# Proyecto de Predicción de Riesgo Crediticio - Módulo 2

Este repositorio contiene la solución a la Práctica 3 del Módulo 2, enfocada en la predicción de impagos de tarjetas de crédito utilizando modelos de clasificación lineal y técnicas de balanceo de datos.

##  Autor
* **Nombre:** Eduardo Castillo García
* **Grupo:** G33

##  Descripción del Problema
El objetivo es predecir la probabilidad de que un cliente caiga en incumplimiento de pago el próximo mes (`default_payment_next_month`). El dataset presenta un desbalanceo de clases significativo, el cual fue abordado mediante técnicas de sobremuestreo y pesos de clase.

## 🛠️ Pipeline de Machine Learning
El modelo final se consolidó en un `Pipeline` de la librería `imblearn` que incluye:

1.  **Preprocesamiento:**
    * Normalización de nombres de columnas (limpieza de caracteres especiales y espacios).
    * Escalamiento de variables financieras con `RobustScaler` para mitigar el impacto de outliers.
2.  **Selección de Variables:**
    * Uso de `SelectFromModel` con un `RandomForestClassifier` para identificar las características con mayor poder predictivo.
3.  **Balanceo de Clases:**
    * Implementación de **SMOTE** (Synthetic Minority Over-sampling Technique) para equilibrar la clase minoritaria durante el entrenamiento.
4.  **Modelo Final:**
    * **Algoritmo:** `SGDClassifier` (Stochastic Gradient Descent).
    * **Configuración:** `penalty='elasticnet'`, `class_weight='balanced'`.
    * **Calibración:** Se utilizó `CalibratedClassifierCV` (método Sigmoide) para permitir que el modelo genere probabilidades (`y_hat`) a pesar de usar una función de pérdida `hinge`.

##  Archivos del Entregable
* `CDD_G33_M2_P3_Eduardo_CastilloGarcia.pkl`: Pipeline completo entrenado y serializado.
* `CDD_G33_M2_P3_Eduardo_CastilloGarcia.csv`: Predicciones de probabilidad (`y_hat`) para el conjunto de test.
* `requirements.txt`: Listado de librerías y versiones necesarias para la ejecución.
* `README.md`: Documentación del proyecto.

##  Instalación y Ejecución
Para replicar el entorno y ejecutar el modelo, instale las dependencias:

### bash
pip install -r requirements.txt