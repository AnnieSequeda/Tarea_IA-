import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def generar_caso_de_uso_preparar_datos():
    """
    Genera un caso de uso aleatorio para una funcion con firma equivalente a:
    calcular_estabilidad_importancia(df, target_col, n_iteraciones)

    Basado en el enunciado adjunto, el output esperado se calcula como:
    std(feature_importances_) a lo largo de varias iteraciones de entrenamiento
    con diferentes random_state.

    Returns:
        tuple[dict, np.ndarray]:
            - input: diccionario con claves 'df', 'target_col', 'n_iteraciones'
            - output: desviacion estandar de importancias por variable
    """
    rng = np.random.default_rng()

    n_muestras = int(rng.integers(80, 251))
    n_features = int(rng.integers(3, 9))
    n_iteraciones = int(rng.integers(5, 26))

    x = rng.normal(loc=0.0, scale=1.2, size=(n_muestras, n_features))

    pesos = rng.normal(loc=0.0, scale=2.0, size=n_features)
    y_lineal = x @ pesos
    y_no_lineal = 0.6 * np.sin(x[:, 0]) + 0.4 * (x[:, 1] ** 2 if n_features > 1 else 0.0)
    ruido = rng.normal(loc=0.0, scale=0.5, size=n_muestras)
    y = y_lineal + y_no_lineal + ruido

    feature_cols = [f"x{i + 1}" for i in range(n_features)]
    target_col = "target"

    df = pd.DataFrame(x, columns=feature_cols)
    df[target_col] = y

    importancias = []
    for semilla in range(n_iteraciones):
        modelo = RandomForestRegressor(
            n_estimators=120,
            random_state=semilla,
            n_jobs=-1,
        )
        modelo.fit(df[feature_cols], df[target_col])
        importancias.append(modelo.feature_importances_)

    output = np.std(np.array(importancias), axis=0)

    input_data = {
        "df": df,
        "target_col": target_col,
        "n_iteraciones": n_iteraciones,
    }

    return input_data, output
