# practico-programacion-avanzada


Scikit-Learn es una de las librerías de aprendizaje automático más populares en Python. Permite a los desarrolladores y científicos de datos implementar una amplia gama de algoritmos de machine learning de una manera sencilla y eficiente.

Uno de los temas que puedo cubrir en detalle es la Regresión Lineal, que es uno de los algoritmos de aprendizaje supervisado más utilizados en Scikit-Learn.

La Regresión Lineal es un modelo que intenta encontrar la mejor línea recta que se ajusta a un conjunto de datos. Esta línea recta se utiliza para predecir valores de una variable dependiente a partir de una o más variables independientes.

Vamos a desarrollar un ejemplo paso a paso:

Importar las librerías necesarias:

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
Generar datos de ejemplo:
Generar datos aleatorios

X = np.random.rand(100, 1)
y = 2 * X + 3 + np.random.normal(0, 0.5, (100, 1))
En este ejemplo, generamos 100 valores aleatorios para la variable independiente X, y luego calculamos la variable dependiente y usando la ecuación
𝑦 = 2𝑋 + 3
y=2X+3 con un pequeño ruido aleatorio.

Entrenar el modelo de Regresión Lineal:
Crear y entrenar el modelo

model = LinearRegression()
model.fit(X, y)
Aquí creamos una instancia del modelo de Regresión Lineal y lo entrenamos con los datos X y y.

Hacer predicciones:
Hacer predicciones

y_pred = model.predict(X)
Utilizamos el modelo entrenado para hacer predicciones sobre los mismos datos de entrada X.

Evaluar el rendimiento del modelo:
Calcular el coeficiente de determinación (R-squared)

r_squared = model.score(X, y)
print(f"R-squared: {r_squared:.2f}")

Calculamos el coeficiente de determinación (R-squared) para evaluar qué tan bien se ajusta el modelo a los datos.

Visualizar los resultados:
Visualizar los datos y la línea de regresión

plt.scatter(X, y, label="Datos reales")
plt.plot(X, y_pred, color="red", label="Predicciones")
plt.xlabel("Variable independiente (X)")
plt.ylabel("Variable dependiente (y)")
plt.title("Regresión Lineal")
plt.legend()
plt.show()

Finalmente, graficamos los datos reales y las predicciones del modelo para visualizar los resultados.

Este es un ejemplo básico de cómo utilizar la Regresión Lineal con Scikit-Learn.
