#  Proyecto de Análisis de Datos y Visualización



Este proyecto contiene funciones para análisis exploratorio de datos (EDA) y visualización, incluyendo detección de outliers, histogramas y gráficos de barras. Está diseñado para facilitar el análisis de DataFrames de pandas y generar gráficos claros con Matplotlib y Seaborn.

Repositorio: [GitHub](https://github.com/LaloCastillo20/Diplomado_DataSciene.git)

---

## 📋 Tabla de Contenidos

- [Funcionalidades](#funcionalidades)
- [Instalación](#instalación)
- [Requierements](#Requirements)
- [Contacto](#contacto)

---

## ⚙️ Funcionalidades

- `calcular_IQR(df, columna)` → Calcula el IQR de una columna numérica y devuelve los límites inferior y superior para detectar outliers.
- `plot_histogram(df, col, group=None, bins=30, kde=True, figsize=(10,8))` → Genera histogramas de columnas numéricas, con opción de KDE y agrupamiento.
- `plot_horizontal_bar(df, col, figsize=(8,5))` → Genera gráficos de barras horizontales ordenadas por frecuencia.
- `check_data_completeness_nombrecompleto(df)` → Analiza la completitud de un DataFrame, muestra nulos, estadísticas básicas y clasifica columnas en continuas o discretas.

---

## 💻 Instalación

1. Clonar el repositorio:

```bash
git clone https://github.com/LaloCastillo20/Diplomado_DataSciene.git
cd Diplomado_DataSciene
```
## 💻 Requirements

```bash
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.2
pytest>=7.3.0
seaborn>=0.12.2
scikit-learn>=1.3.0
```
## 💻 contacto
1.- Eduardo Castillo Garcia  email: 19castillog20@gmail.com
