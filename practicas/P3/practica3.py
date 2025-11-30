import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import seaborn as sns
from practicas.P3 import plots
import os 
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath('.')))
sys.path.append(project_root)

def preprocesamiento_datos (ruta:str):
   
    """
    Preprocesamiento completo de un dataset CTG a partir de un archivo CSV.

    Esta función realiza las siguientes operaciones:
    1. Lectura del CSV y limpieza de columnas de control ('FileName', 'Date', 'SegFile').
    2. Normalización de los nombres de columnas.
    3. Identificación de columnas numéricas y cálculo de su varianza.
    4. Visualización de histogramas para columnas numéricas.
    5. Visualización de conteos de valores para columnas categóricas seleccionadas.
    6. Eliminación de columnas con varianza baja (por debajo del percentil 10%).
    7. Imputación de valores faltantes:
       - Media para columnas continuas.
       - Mediana para columnas discretas o categóricas.
    8. Detección de outliers mediante IQR y eliminación de valores fuera de los límites.
    9. Retorno del DataFrame limpio y listo para análisis.

    Parámetros
    ----------
    ruta : path
        Ruta al archivo CSV que contiene los datos a procesar.

    RetornS
    -------
    df_sin_outliers : pandas.DataFrame
        DataFrame preprocesado con valores faltantes imputados, columnas de baja varianza eliminadas
        y outliers eliminados para columnas seleccionadas.

    """

    df=pd.read_csv(ruta)

    df.columns = df.columns.str.strip()

    df.describe()

    ### Eliminamos las columnas de control 

    df.drop(columns=['FileName','Date','SegFile'],inplace=True)
    # Seleccionar columnas numéricas
    num_cols = df.select_dtypes(include=['int64', 'float64'])

    # Calcular varianza
    varianzas = num_cols.var()


    num_cols = df.select_dtypes(include=['int64', 'float64'])

    # Graficar histogramas para todas las columnas
    num_cols.hist(figsize=(15, 10), bins=30)
    plt.tight_layout()
    plt.show()

    lista=['FM','DS','DP','DR','C','D','E','AD','LD','FS','SUSP']

    for col in lista:
        print(f"\nDistribución de {col}:")
        print(df[col].value_counts())

    ### Eliminamos aquellas que se encuentren del 10% de la varianza
    umbral = varianzas.quantile(0.10)
    cols_a_eliminar = varianzas[varianzas <= umbral].index
    df.drop(columns=cols_a_eliminar,inplace=True)

    print(f'Las columnas a eliminar son: {cols_a_eliminar}')
    
    num_cols = df.select_dtypes(include=['int64', 'float64'])

    # Graficar histogramas para todas las columnas
    num_cols.hist(figsize=(15, 10), bins=30)
    plt.tight_layout()
    plt.show()

    

    df.info()

    ### Separamos el Dataset para aplicar ambos metodos 


    media_cols = ['b', 'e', 'LBE', 'LB', 'AC', 'FM', 'UC', 'ASTV', 'MSTV', 'ALTV', 'MLTV',
                'DL', 'DP', 'Width', 'Min', 'Max', 'Nmax']

    mediana_cols = ['Nzeros', 'Mode', 'Mean','Median', 'Variance', 'Tendency', 'A', 'B', 'AD', 'DE', 'LD', 'SUSP','CLASS', 'NSP']

    imputer_mean = SimpleImputer(strategy='mean')

    df[media_cols] = pd.DataFrame( imputer_mean.fit_transform(df[media_cols]), columns=media_cols,index=df.index )

    imputer_median = SimpleImputer(strategy='median')

    df[mediana_cols] = pd.DataFrame(imputer_median.fit_transform(df[mediana_cols]),  columns=mediana_cols,index=df.index)


    # Deteccion de Outliers

    ### Evitamos columnas binarias
    cols_a_eliminar = varianzas[varianzas >= umbral].index
    cols_to_check = [col for col in cols_a_eliminar if df[col].nunique() > 5]

    # Graficar boxplots solo de esas columnas
    for col in cols_to_check:

        plots.plot_histogram(df=df, col=col, kde=True, bins=30, figsize=(8,4))
    


    inferior,superior = calcular_IQR(df, 'b')

    df_sin_outliers = df[(df['b'] >= inferior) & (df['b'] <= superior)]
    inferior,superior = calcular_IQR(df, 'FM')
    df_sin_outliers = df[(df['FM'] >= inferior) & (df['FM'] <= superior)]

    inferior,superior = calcular_IQR(df, 'ALTV')
    df_sin_outliers = df[(df['ALTV'] >= inferior) & (df['ALTV'] <= superior)]
    inferior,superior = calcular_IQR(df, 'Variance')
    df_sin_outliers = df[(df['Variance'] >= inferior) & (df['Variance'] <= superior)]
    
    ### Graficamos el HeatMap
    plots.plot_heatmap(df=df_sin_outliers, cols=None, method=1, figsize=(12,10))

    ### Graficamos sin outliers (Violin)
    for col in cols_to_check :
        plots.plot_violin(df_sin_outliers, col=col)

    ### Grafico de Densidad
    for col in cols_to_check :
        plots.plot_densidad(df_sin_outliers, col=col)


    return df_sin_outliers



#### Funciones Auxiliares


def calcular_IQR(df:pd.DataFrame, columna:str ):

    """ 
    Descripcion:
        Calcula el IQR y detecta outliers para una columna numérica del dataframe dado.
    
    Parámetros:
        df : DataFrame
            DataFrame que contiene la columna.
        columna : str
            Nombre de la columna numérica.

    Returna:
    
        - límite inferior y  limite superior
    """
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    return limite_inferior,limite_superior
        
