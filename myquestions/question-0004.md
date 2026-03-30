# El Problema: Análisis de la Evolución del Rendimiento mediante Selección Recursiva (RFE)

En ingeniería de características, no siempre "más es mejor". A veces, añadir variables de baja calidad introduce ruido y degrada el modelo.

La Eliminación Recursiva de Características (RFE) es una técnica que clasifica las variables según su importancia y permite observar cómo mejora (o empeora) el modelo a medida que añadimos las mejores variables una a una.

El reto consiste en generar una curva de rendimiento que muestre el coeficiente de determinación ($R^2$) a medida que el modelo utiliza desde la variable más importante hasta completar el conjunto de datos.

## Tu Misión

Escribe una función que utilice RFE para rankear las variables y luego calcule el desempeño incremental del modelo.

## Nombre de la función

`medir_rendimiento_incremental(df, target_col)`

## Argumentos

- `df` (`pandas.DataFrame`): DataFrame con variables predictoras numéricas y una variable objetivo continua.
- `target_col` (`str`): Nombre de la columna objetivo.

## Retorno esperado

- `numpy.ndarray`: Arreglo con los valores de $R^2$ obtenidos al entrenar el modelo con las mejores $n$ características, donde $n$ va desde 1 hasta el número total de variables predictoras.