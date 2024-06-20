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

introduccion

vamos a realizar una investigación utilizando el método de Regresión Lineal con un enfoque en aplicaciones útiles para la sociedad.

Tema: Predicción del consumo de energía eléctrica en hogares

Contexto:
El consumo eficiente de energía eléctrica es un tema relevante para la sociedad, ya que puede ayudar a reducir el impacto ambiental,
disminuir los costos energéticos y promover un uso más sostenible de los recursos. Comprender los factores que influyen en el consumo
de electricidad en los hogares es clave para desarrollar estrategias de ahorro y eficiencia energética.

Objetivo:
Utilizar un modelo de Regresión Lineal para predecir el consumo de energía eléctrica en los hogares, 
identificando las variables más influyentes.

Datos:
Vamos a utilizar un conjunto de datos públicos que contiene información sobre el consumo de energía eléctrica en hogares,
así como características como el tamaño de la vivienda, el número de ocupantes, el tipo de electrodomésticos, etc.

Pasos:

Recopilar y limpiar los datos necesarios.
Explorar y analizar las variables que pueden influir en el consumo de energía.
Entrenar un modelo de Regresión Lineal utilizando Scikit-Learn.
Evaluar el rendimiento del modelo y su capacidad predictiva.
Identificar las variables más importantes que afectan el consumo de energía.
Interpretar los resultados y discutir las implicaciones prácticas.
Proponer recomendaciones y estrategias para mejorar la eficiencia energética en los hogares.
Beneficios para la sociedad:

Ayudar a los usuarios a comprender mejor sus patrones de consumo de energía y tomar medidas para reducir el gasto.
Permitir a las empresas de servicios públicos y a los gobiernos desarrollar programas y políticas más efectivas para promover el ahorro de energía.
Contribuir a la reducción del consumo de energía y las emisiones de gases de efecto invernadero, lo que beneficia al medioambiente.
Generar ahorros económicos para los hogares al optimizar el uso de la energía eléctrica.

segundo paso generar los algoritmos necesarios para continuar con el proyecto

Importar las librerías necesarias:
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
Cargar y preparar los datos:

# Cargar los datos desde un archivo CSV
data = np.genfromtxt('datos_consumo_energia.csv', delimiter=',')

# Separar las variables independientes (X) y la variable dependiente (y)
X = data[:, :-1]  # Todas las columnas excepto la última
y = data[:, -1]   # Última columna (consumo de energía)
Dividir los datos en conjuntos de entrenamiento y prueba:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Entrenar el modelo de Regresión Lineal:

model = LinearRegression()
model.fit(X_train, y_train)
Hacer predicciones y evaluar el rendimiento:



# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular el coeficiente de determinación (R-squared)
r_squared = model.score(X_test, y_test)
print(f"R-squared: {r_squared:.2f}")
Visualizar los resultados:


# Visualizar los datos y las predicciones
plt.scatter(y_test, y_pred, label="Predicciones")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label="Línea ideal")
plt.xlabel("Consumo de energía real")
plt.ylabel("Consumo de energía predicho")
plt.title("Regresión Lineal: Predicción del consumo de energía")
plt.legend()
plt.show()
Este es un ejemplo básico de cómo utilizar Scikit-Learn para generar un modelo de Regresión Lineal para predecir el consumo de energía eléctrica en hogares.

Algunas extensiones y mejoras que podrías considerar:


