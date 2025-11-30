import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_histogram(df:pd.DataFrame, col:str, group=None, bins=30, kde=True, figsize=(10,8 )):
    """

    Descripcion:
    Crea un histograma de una variable numérica con opción de añadir KDE.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame que contiene los datos.
    col : str
        Nombre de la columna a graficar.
    group : str, opcional
        Columna categórica para graficar grupos separados (default None).
    bins : int
        Número de bins del histograma (default 30).
    kde : bool
        Si True, añade línea de densidad (default True).
    figsize : tuple
        Tamaño de la figura (default (8,5)).
    """
    plt.figure(figsize=figsize)
    if group:
        for g in df[group].unique():
            subset = df[df[group]==g]
            sns.histplot(subset[col], bins=bins, kde=kde, label=str(g), stat='density', alpha=0.5)
        plt.legend(title=group)
    else:
        sns.histplot(df[col], bins=bins, kde=kde, stat='density')
    plt.title(f'Histograma de {col}')
    plt.show()

def plot_horizontal_bar(df, col, figsize=(8,5)):
    """
    Descripcion:
    Crea un gráfico de barras horizontales ordenadas por frecuencia descendente.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame que contiene los datos.
    col : str
        Columna categórica a graficar.
    figsize : tuple
        Tamaño de la figura (default (8,5)).
    """
    counts = df[col].value_counts().sort_values(ascending=True)
    counts.plot(kind='barh', figsize=figsize)
    plt.xlabel('Frecuencia')
    plt.ylabel(col)
    plt.title(f'Barras horizontales de {col}')
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

def plot_violin(df:pd.DataFrame, col:str, figsize=(8,5)):
    """
    Crea un gráfico Violín con overlay de Swarmplot para visualizar
    la distribución de una variable numérica, opcionalmente por grupo.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene los datos.
    col : str
        Nombre de la columna numérica a graficar.
    group : str, opcional
        Nombre de la columna categórica que define los grupos.
        Si None, se grafica toda la columna como un solo grupo.
    figsize : tuple
        Tamaño de la figura (ancho, alto). Default (8,5).

    """
    plt.figure(figsize=figsize)
    
   
    # Violín sin grupo
    sns.violinplot(y=col, data=df, inner=None, color='lightgray')
    sns.swarmplot(y=col, data=df, color='k', alpha=0.6)
    
    plt.ylabel(col)
    plt.title(f'Violín + Swarm de {col}' )
    plt.show()


def plot_heatmap(df:pd.DataFrame, cols=None, method=1, figsize=(10,8)):
    """
    Crea un heatmap de correlación entre columnas numéricas.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos.
    cols : list o None
        Columnas a incluir en la correlación. Si None, se usan todas numéricas.
    method : int
        Método de correlación ('1:pearson', '2: spearman').
    figsize : tuple
        Tamaño de la figura (default (10,8)).
    """
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns
    method_map = {1: 'pearson', 2: 'spearman'}
    corr =  corr = df[cols].corr(method=method_map[method])
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Heatmap de correlación ({method})')
    plt.show()


def plot_densidad(df:pd.DataFrame, col=None, group=None, figsize=(8,5), palette='tab10', fill=True):
    """
    Crea un gráfico de densidad (KDE) 

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame con los datos.
    col : str
        Nombre de la columna numérica a graficar.
    group : str, opcional
        Columna categórica para diferenciar curvas por color. Si None, se grafica solo una curva.
    figsize : tuple
        Tamaño de la figura (default (8,5)).
    palette : str o lista
        Paleta de colores para las clases (default 'tab10').
    fill : bool
        Si True, rellena el área bajo la curva (default True).

    Notas
    -----
    - Cada clase se dibuja con un color diferente si se proporciona `group`.
    - Ideal para comparar distribuciones de varias categorías.
    """
    plt.figure(figsize=figsize)
    sns.kdeplot(df[col], fill=fill, color='blue', label=col)
    plt.legend()
    
    plt.title(f'Densidad de {col}' + (f' por {group}' if group else ''))
    plt.xlabel(col)
    plt.ylabel('Densidad')
    plt.show()




