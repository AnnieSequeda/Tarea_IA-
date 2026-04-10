import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def generar_caso_de_uso_Optimizacion():
    """
    Genera un caso de uso aleatorio para la función evaluar_umbrales_decision.
    
    Basado en el enunciado adjunto, la función evaluar_umbrales_decision debe:
    - Entrenar un modelo de clasificación binaria
    - Extraer probabilidades de la clase positiva
    - Calcular F1-Score para 9 umbrales diferentes
    
    Returns:
        tuple[dict, np.ndarray]:
            - input: diccionario con claves 'df', 'target_col'
            - output: array de 9 valores de F1-Score, uno para cada umbral
    """
    rng = np.random.default_rng()
    
    # Generar número aleatorio de muestras y características
    n_muestras = int(rng.integers(50, 201))
    n_features = int(rng.integers(3, 8))
    
    # Matriz de predictores aleatoria
    x = rng.normal(loc=0.0, scale=1.5, size=(n_muestras, n_features))
    
    # Crear variable objetivo binaria de forma que sea realista
    # Usamos una combinación lineal con una función no lineal para crear separabilidad
    pesos_reales = rng.normal(loc=0.0, scale=1.0, size=n_features)
    score = x @ pesos_reales
    # Aplicamos logit para convertir a probabilidades
    probabilidades = 1 / (1 + np.exp(-score))
    y = (probabilidades + rng.normal(loc=0.0, scale=0.1, size=n_muestras)) > 0.5
    y = y.astype(int)
    
    # Asegurar que tenemos ambas clases
    while np.sum(y) < 2 or np.sum(y) > n_muestras - 2:
        probabilidades = 1 / (1 + np.exp(-rng.normal(loc=0.0, scale=1.0, size=n_features) @ x.T))
        y = (probabilidades + rng.normal(loc=0.0, scale=0.1, size=n_muestras)) > 0.5
        y = y.astype(int)
    
    # Crear DataFrame
    feature_cols = [f"x{i + 1}" for i in range(n_features)]
    target_col = "target"
    
    df = pd.DataFrame(x, columns=feature_cols)
    df[target_col] = y
    
    # Entrenar modelo de clasificación logística
    modelo = LogisticRegression(random_state=42, max_iter=1000)
    modelo.fit(df[feature_cols], df[target_col])
    
    # Obtener probabilidades de la clase positiva
    y_proba = modelo.predict_proba(df[feature_cols])[:, 1]
    
    # Calcular F1-Score para 9 umbrales uniformemente espaciados de 0.1 a 0.9
    umbrales = np.linspace(0.1, 0.9, 9)
    f1_scores = []
    
    for umbral in umbrales:
        y_pred = (y_proba >= umbral).astype(int)
        # Evitar warning si todas las predicciones son de una clase
        if len(np.unique(y_pred)) > 1:
            f1 = f1_score(df[target_col], y_pred)
        else:
            # Si solo hay una clase predicha, F1 es 0 o indefinido
            f1 = 0.0 if np.all(y_pred == 0) else 0.0
        f1_scores.append(f1)
    
    output = np.array(f1_scores)
    
    input_data = {
        "df": df,
        "target_col": target_col,
    }
    
    return input_data, output
