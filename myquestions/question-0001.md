El Problema: Evaluación del Impacto de Regularización L1 vs L2
En el modelado predictivo de Machine Learning, es fundamental comprender cómo las distintas penalizaciones (regularización) afectan los pesos o la importancia que el modelo le asigna a las variables. La regularización L1 (Lasso) tiende a reducir los coeficientes a cero (creando modelos dispersos), mientras que L2 (Ridge) encoge los coeficientes pero rara vez los elimina por completo.

El problema consiste en medir numéricamente la divergencia exacta entre los pesos asignados por ambos métodos bajo un mismo nivel de penalización.

Tu Misión
Debes desarrollar una función que procese un conjunto de datos, aplique ambos tipos de regularización y calcule la diferencia absoluta entre los coeficientes resultantes de cada modelo.

Nombre de la función: comparar_penalizaciones(df, target_col, alpha)

Argumentos:

df (pandas.DataFrame): Un DataFrame que contiene las variables predictoras (numéricas) y la variable objetivo.

target_col (str): El nombre de la columna dentro del DataFrame que actúa como variable objetivo (y).

alpha (float): El valor del hiperparámetro de penalización que se aplicará a ambos modelos.

Retorno esperado:

numpy.ndarray: Un arreglo unidimensional (vector) de Python que contenga la diferencia absoluta entre los coeficientes del modelo Lasso y los coeficientes del modelo Ridge.