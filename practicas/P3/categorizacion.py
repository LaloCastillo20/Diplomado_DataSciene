
import pandas as pd

def check_data_completeness_nombrecompleto(df:pd.DataFrame):
    """
     Descripcion:

        Da una descripcion de como esta estrucuturado nuestro df, conteo de nulos, datos estadisticos, y clasificacion de las columnas 
        en columnas discretas o continuas
    Args:
        df: Dataframe para analizar
    
    Retorna:
        resumen: Datafrme con el analisis 
    """
    resumen = pd.DataFrame()
    

    ## Procesamiento de Datos 
    resumen['num_nulos'] = df.isna().sum()
    resumen['completitud_%'] = (1 - df.isna().sum() / len(df)) * 100
    resumen['dtype'] = df.dtypes
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    resumen_dispersion = df[num_cols].describe().T[['min', 'max', 'mean', 'std']]
    resumen = resumen.join(resumen_dispersion, how='left')
    
    ## Clasificacion de Columnas
    resumen['tipo_columna'] = [
        'Continua' if (
            pd.api.types.is_numeric_dtype(df[col]) 
            and df[col].nunique() > 10
        ) else 'Discreta'
        for col in df.columns
    ]
    
    return resumen