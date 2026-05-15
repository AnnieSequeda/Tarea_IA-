import pandas as pd


def detectar_ocupacion_fantasma(
    sensor_data,
    pagos_activos
):

    ocupadas = sensor_data[
        sensor_data['estado_sensor'] == 1
    ]['celda_id']

    pagadas = set(
        pagos_activos['celda_id']
    )

    fantasmas = sorted([
        celda for celda in ocupadas
        if celda not in pagadas
    ])

    return fantasmas
