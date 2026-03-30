import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def generar_caso_de_uso_preparar_datos():
    """
    Genera un caso de uso aleatorio para una funcion con firma equivalente a:
    medir_rendimiento_incremental(df, target_col)

    Basado en el enunciado adjunto, el output esperado se calcula como:
    curva de R^2 incremental usando el ranking de variables obtenido con RFE.

    Returns:
        tuple[dict, np.ndarray]:
            - input: diccionario con claves 'df', 'target_col'
            - output: array con R^2 para n = 1..numero de variables
    """
    rng = np.random.default_rng()

    n_muestras = int(rng.integers(60, 181))
    n_features = int(rng.integers(3, 10))

    # Predictores numericos con escalas distintas para hacer el caso mas realista.
    x = rng.normal(loc=0.0, scale=1.0, size=(n_muestras, n_features))
    x = x * rng.uniform(0.5, 2.5, size=(1, n_features))

    # Objetivo continuo con senal dominante en algunas variables y ruido moderado.
    pesos = rng.normal(loc=0.0, scale=2.0, size=n_features)
    pesos[rng.random(n_features) < 0.35] = 0.0
    ruido = rng.normal(loc=0.0, scale=0.8, size=n_muestras)
    y = x @ pesos + ruido

    feature_cols = [f"x{i + 1}" for i in range(n_features)]
    target_col = "target"

    df = pd.DataFrame(x, columns=feature_cols)
    df[target_col] = y

    x_df = df[feature_cols]
    y_sr = df[target_col]

    selector = RFE(estimator=LinearRegression(), n_features_to_select=1, step=1)
    selector.fit(x_df, y_sr)

    ranking = selector.ranking_
    orden_idx = np.argsort(ranking)
    mejores_cols = [feature_cols[i] for i in orden_idx]

    r2_incremental = []
    for n in range(1, len(mejores_cols) + 1):
        cols_n = mejores_cols[:n]
        modelo = LinearRegression()
        modelo.fit(x_df[cols_n], y_sr)
        pred = modelo.predict(x_df[cols_n])
        r2_incremental.append(r2_score(y_sr, pred))

    output = np.array(r2_incremental)

    input_data = {
        "df": df,
        "target_col": target_col,
    }

    return input_data, output
