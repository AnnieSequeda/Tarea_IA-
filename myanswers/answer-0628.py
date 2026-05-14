import pandas as pd


def detectar_ocupacion_fantasma(sensor_data=None):

    if sensor_data is None:
        return None

    df_sensores = sensor_data['sensor_data']
    df_pagos = sensor_data['pagos_activos']

    ocupadas = df_sensores[
        df_sensores['estado_sensor'] == 1
    ]['celda_id']

    pagadas = set(df_pagos['celda_id'])

    fantasmas = sorted([
        celda for celda in ocupadas
        if celda not in pagadas
    ])

    return fantasmas
