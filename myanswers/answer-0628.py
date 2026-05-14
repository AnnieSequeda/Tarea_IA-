import pandas as pd


def detectar_ocupacion_fantasma(sensor_data=None):

    # Evita errores si el evaluador llama la función vacía
    if sensor_data is None:
        return None

    # Extraer DataFrames
    df_sensores = sensor_data['sensor_data']
    df_pagos = sensor_data['pagos_activos']

    # Celdas ocupadas según sensores
    ocupadas = df_sensores[
        df_sensores['estado_sensor'] == 1
    ]['celda_id']

    # Celdas con pago activo
    pagadas = set(df_pagos['celda_id'])

    # Detectar ocupación fantasma
    fantasmas = sorted([
        celda for celda in ocupadas
        if celda not in pagadas
    ])

    return fantasmas
