import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


def agrupar_zonas_entrega_dbscan(X, epsilon, min_muestras):

    # Escalar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Crear modelo DBSCAN
    dbscan = DBSCAN(
        eps=epsilon,
        min_samples=min_muestras
    )

    # Ajustar y obtener etiquetas
    etiquetas = dbscan.fit_predict(X_scaled)

    return etiquetas
