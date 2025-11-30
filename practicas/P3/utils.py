
import pandas as pd

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
        
