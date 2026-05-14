import pandas as pd


def detectar_ocupacion_fantasma(data):

    sensor_data = data["sensor_data"]
    pagos_activos = data["pagos_activos"]

    # Celdas marcadas como ocupadas
    ocupadas = sensor_data[
        sensor_data["estado_sensor"] == 1
    ]["celda_id"]

    # Celdas con pago
    pagadas = set(pagos_activos["celda_id"])

    # Detectar fantasmas
    fantasmas = sorted([
        celda for celda in ocupadas
        if celda not in pagadas
    ])

    return fantasmas
