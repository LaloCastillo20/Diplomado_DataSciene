
import pandas as pd
from varclushi import VarClusHi
import plotly.express as px
import numpy as np

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
        


def calcular_woe_iv(df:pd.DataFrame, target:str):
    """
    Descripción: 
        Calcula Weight of Evidence (WoE) e Information Value (IV) para todas 
        las variables categóricas en un DataFrame.

    Parámetros

    df : pandas.DataFrame
            Conjunto de datos que contiene variables categóricas y la variable objetivo.
    target : str
            Nombre de la columna objetivo binaria (0 = bueno, 1 = malo).

    Retorno

    woe_df : pandas.DataFrame
        DataFrame con columnas: ['variable', 'categoria', 'WoE'] 

    iv_df : pandas.DataFrame
        DataFrame con columnas: ['variable', 'IV']
    """
    cat_vars = df.select_dtypes(include=['object', 'category']).columns
    ## num_vars = df.columns.drop('target')
    
    woe_rows = []   
    iv_rows = []   
    for var in cat_vars:
     
        tbl = df.groupby(var)[target].agg(['count', 'sum'])
        tbl.columns = ['total', 'malos']
        tbl['buenos'] = tbl['total'] - tbl['malos']

        pct_b = tbl['buenos'] / tbl['buenos'].sum()
        pct_m = tbl['malos']  / tbl['malos'].sum()

        tbl['WoE']  = np.log(pct_b / pct_m)
        tbl['IV_i'] = (pct_b - pct_m) * tbl['WoE']

        for categoria, fila in tbl.iterrows():
            woe_rows.append([var, categoria, fila['WoE']])

        iv_rows.append([var, tbl['IV_i'].sum()])


    woe_df = pd.DataFrame(woe_rows, columns=['variable', 'categoria', 'WoE'])

    iv_df = pd.DataFrame(iv_rows, columns=['variable', 'IV']).sort_values('IV', ascending=False)

    return woe_df, iv_df
def analizar_varclushi(dataframe, features_list, max_eigval2=1):
    """
    Ejecuta VarClusHi y grafica los resultados para seleccionar variables.
    El varclushi agrupa variables correlacionadas en clusters y selecciona
    la mejor variable representante de cada cluster.
    Valores RS_Ratio bajos indican buenas variables líderes.
    """
    vc = VarClusHi(dataframe[features_list], maxeigval2=max_eigval2, maxclus=None)
    vc.varclus()
    
    resumen = vc.rsquare
    
    # 2. Seleccionar la mejor variable de cada cluster (la del Ratio más bajo)
    # El (1-RS_Own) bajo indica que se explica bien por su propio cluster
    # El (1-RS_Next) alto indica que NO se explica por el siguiente cluster
    # Por tanto, buscamos minimizar el RS_Ratio = (1-RS_Own)/(1-RS_Next)
    
    best_vars = resumen.sort_values('RS_Ratio').groupby('Cluster').first()
    
    print(f"--- VarClusHi: Se encontraron {len(best_vars)} clusters ---")
    print(f"Variables seleccionadas (Líderes): {best_vars['Variable'].tolist()}")
    
    fig = px.bar(
        resumen, 
        x='Cluster', 
        y='RS_Ratio', 
        color='Variable',
        barmode='group',
        title='VarClusHi: Selección de Variables (Barra más baja = Mejor Representante)',
        hover_data=['RS_Own', 'RS_NC']
    )
    
    fig.add_hline(y=0.5, line_dash="dot", annotation_text="Zona de Alta Calidad")
    fig.show()
    
    return resumen, best_vars['Variable'].tolist()
