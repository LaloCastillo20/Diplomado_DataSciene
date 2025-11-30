
import numpy as np
import pandas as pd 

####################### SelectKBest ########################

""" 
SelectKBest es un método de selección de características univariado, parte de la librería scikit-learn.
Su función principal es identificar y conservar las K variables más predictivas según una función de puntuación estadística.


Funcionamiento interno

--> Evaluación univariada

    Cada variable independiente se evalúa por separado.
    La evaluación se hace mediante una función estadística que mide su relación con la variable objetivo (y).

La puntuación puede ser:

        ---> F-value (ANOVA) → Diferencias medias entre grupos (clasificación)

        ---> Chi-cuadrado (χ²) → Dependencia estadística para variables discretas no negativas

        ---> Información mutua → Medida de dependencia no lineal entre variable y objetivo

        ---> Regresión F → Para problemas de regresión

    Ordenamiento de variables

        ---> Cada variable obtiene un puntaje (score) y, opcionalmente, un valor p.

        ---> Las variables se ordenan de mayor a menor puntuación.

Selección de las K mejores

        --> Se seleccionan solo las K variables con mejor score.

        --> k puede ser un número fijo (k=5) o un porcentaje del total (k='all' para conservar todo).

"""














####################### IV y WoE #################
""" 
Peso de Evidencia (WoE)
El Weight of Evidence (WoE) es una transformación que mide la evidencia que un grupo (bin) aporta para distinguir entre dos clases, típicamente "buenos" y "malos".
Se define como:

        WoE_i = ln( (%Buenos_i) / (%Malos_i) )

Donde:
    %Buenos_i = Buenos_i / Total_Buenos
    %Malos_i = Malos_i / Total_Malos

Interpretación:

    WoE > 0: el bin contiene proporcionalmente más buenos.

    WoE < 0: el bin contiene proporcionalmente más malos.

    WoE = 0: comportamiento promedio.

Valor de Información (IV)

    El Information Value (IV) mide la capacidad predictiva de una variable para diferenciar las dos clases.
    Se calcula como:

        IV = Σ [ (%Buenos_i - %Malos_i) * WoE_i ]

Interpretación típica del IV:

        < 0.02 : no útil

        0.02 --> 0.1"  débil

        0.1 --> 0.3 : medio

        0.3 --> 0.5 : fuerte

        0.5 : posible sobreajuste o separación casi perfecta
"""



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
