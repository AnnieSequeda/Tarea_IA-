import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score


def analizar_sensores(df, ventana=3):

    # Copia para evitar modificar original
    df = df.copy()

    # Crear volatilidad
    df["Volatilidad"] = (
        df["Temperatura"]
        .rolling(window=ventana)
        .std()
    )

    # Eliminar NaN
    df = df.dropna()

    # Percentil 90
    umbral = np.percentile(df["Volatilidad"], 90)

    # Crear target
    df["y"] = (df["Volatilidad"] > umbral).astype(int)

    # Variables predictoras
    X = df[["Temperatura", "Volatilidad"]]

    # Variable objetivo
    y = df["y"]

    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Modelo
    modelo = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Entrenamiento
    modelo.fit(X_train, y_train)

    # Predicciones
    y_pred = modelo.predict(X_test)

    # Precision score
    precision = precision_score(y_test, y_pred)

    return precision
