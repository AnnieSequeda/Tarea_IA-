import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge


def generar_caso_de_uso_preparar_datos():
    """
    Genera un caso de uso aleatorio para una funcion con firma equivalente a:
    preparar_datos(df, target_col, alpha)   


    Basado en el enunciado adjunto, el output esperado se calcula como:
    abs(coeficientes_lasso - coeficientes_ridge)

    Returns:
        tuple[dict, np.ndarray]:
            - input: diccionario con claves 'df', 'target_col', 'alpha'
            - output: vector con la diferencia absoluta entre coeficientes
    """
    rng = np.random.default_rng()

    n_muestras = int(rng.integers(30, 121))
    n_features = int(rng.integers(3, 8))

    # Matriz de predictores aleatoria
    x = rng.normal(loc=0.0, scale=1.5, size=(n_muestras, n_features))

    # Construimos una y lineal con algo de ruido para evitar casos degenerados
    pesos_reales = rng.normal(loc=0.0, scale=2.0, size=n_features)
    ruido = rng.normal(loc=0.0, scale=0.3, size=n_muestras)
    y = x @ pesos_reales + ruido

    feature_cols = [f"x{i + 1}" for i in range(n_features)]
    target_col = "target"

    df = pd.DataFrame(x, columns=feature_cols)
    df[target_col] = y

    # Alpha positivo para ambos modelos
    alpha = float(rng.uniform(0.01, 3.0))

    lasso = Lasso(alpha=alpha, fit_intercept=True, max_iter=20000)
    ridge = Ridge(alpha=alpha, fit_intercept=True)

    lasso.fit(df[feature_cols], df[target_col])
    ridge.fit(df[feature_cols], df[target_col])

    output = np.abs(lasso.coef_ - ridge.coef_)

    input_data = {
        "df": df,
        "target_col": target_col,
        "alpha": alpha,
    }

    return input_data, output
