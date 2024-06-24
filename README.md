Introducción

En la gestión de residuos urbanos, la predicción precisa de las cantidades recuperadas es crucial para la planificación eficiente y sostenible. En este contexto, el algoritmo de regresión lineal se presenta como una herramienta poderosa para analizar y predecir las toneladas de residuos recuperados en distintos centros de reciclaje y compostaje de la ciudad. Este ensayo justifica la aplicación de este algoritmo, destacando su relevancia, precisión y contribución a la gestión ambiental.
Relevancia del Algoritmo de Regresión Lineal

La regresión lineal es un método estadístico que modela la relación entre una variable dependiente y una o más variables independientes. En el caso de la gestión de residuos, la variable dependiente podría ser las toneladas recuperadas en centros verdes de la ciudad, mientras que las variables independientes podrían incluir las toneladas procesadas en centros de reciclaje y compostaje.

Aplicar un algoritmo de regresión lineal en este contexto es relevante por varias razones:

    Simplicidad y Eficacia: La regresión lineal es fácil de implementar y entender, lo que la convierte en una herramienta accesible para los gestores de residuos. Su eficacia en la identificación de relaciones lineales en los datos permite generar predicciones precisas.
    Identificación de Tendencias: Este algoritmo ayuda a identificar tendencias y patrones en los datos históricos, proporcionando una base sólida para la toma de decisiones futuras.
    Optimización de Recursos: Con predicciones precisas, es posible optimizar la asignación de recursos, mejorando la eficiencia operativa y reduciendo costos.

Precisión en la Predicción

La precisión es un aspecto fundamental en la gestión de residuos. Las predicciones inexactas pueden llevar a la sobrecarga de los centros de procesamiento o a la subutilización de la capacidad instalada. La regresión lineal, al modelar la relación entre variables clave, ofrece una alta precisión en las predicciones debido a las siguientes razones:

    Análisis de Datos Históricos: Utiliza datos históricos para entrenar el modelo, lo que permite captar las dinámicas reales de la recuperación de residuos.
    Coeficientes Ajustados: Los coeficientes obtenidos a partir de la regresión reflejan el impacto de cada variable independiente en la variable dependiente, permitiendo ajustes finos en las predicciones.
    Validación y Ajuste del Modelo: El algoritmo permite validar y ajustar el modelo continuamente, mejorando su precisión con el tiempo a medida que se dispone de más datos.

Contribución a la Gestión Ambiental

El uso del algoritmo de regresión lineal no solo mejora la eficiencia operativa sino que también tiene un impacto positivo en la gestión ambiental:

    Reducción de Residuos en Vertederos: Predicciones precisas facilitan la planificación y operación de los centros de reciclaje y compostaje, reduciendo la cantidad de residuos que terminan en vertederos.
    Mejora en el Reciclaje y Compostaje: Al conocer de antemano las cantidades de residuos, los centros pueden preparar mejor sus operaciones, aumentando las tasas de reciclaje y compostaje.
    Sostenibilidad: Una gestión más eficiente de los residuos contribuye a la sostenibilidad urbana, reduciendo la huella de carbono y promoviendo prácticas más ecológicas.

Implementación Práctica

La implementación del algoritmo de regresión lineal en la gestión de residuos implica varios pasos, que incluyen la carga de datos, el procesamiento y el entrenamiento del modelo, así como la visualización y validación de los resultados. Dividir el proceso en módulos y utilizar un enfoque orientado a objetos mejora la modularidad y reutilización del código, facilitando su mantenimiento y actualización.

Por ejemplo, en un proyecto práctico, los datos se cargan desde un archivo Excel y se procesan para identificar las variables clave. El modelo de regresión se entrena utilizando estos datos, y se realizan predicciones sobre futuras cantidades de residuos recuperados. La visualización de los resultados permite verificar la precisión del modelo y ajustar los parámetros según sea necesario.
Conclusión

La aplicación del algoritmo de regresión lineal en la predicción de toneladas recuperadas en centros de reciclaje y compostaje es una herramienta valiosa para la gestión de residuos. Su relevancia, precisión y contribución a la sostenibilidad ambiental justifican su uso en este contexto. A través de predicciones precisas y análisis de datos históricos, los gestores de residuos pueden optimizar recursos, reducir costos y mejorar la eficiencia operativa, contribuyendo así a un manejo más sostenible de los residuos urbanos.


archivos utilizados

data_loading.py: Este archivo se encargará de la carga de datos.


data_processing.py: Este archivo se encargará del procesamiento y cálculos necesarios.


main.py: Este archivo ejecutará la regresión lineal y generará las visualizaciones.



1.Archivo data_loading.py

import pandas as pd

def load_data(file_path):
    return pd.read_excel(file_path)

2. Archivo data_processing.py

 import numpy as np
from sklearn.linear_model import LinearRegression

def process_data(recuperado):
    x = recuperado[["Centro de Reciclaje de la Ciudad + Centros de Compostaje"]]
    y = recuperado[["Centros Verdes de la Ciudad"]]
    return x, y

def calculate_manual_coefficients(matriz):
    n = len(matriz)
    suma_de_x = np.sum(matriz[:, 1])
    suma_de_y = np.sum(matriz[:, 2])
    suma_de_producto = np.sum(matriz[:, 1] * matriz[:, 2])
    suma_de_cuadrado_x = np.sum(matriz[:, 1] * matriz[:, 1])
    
    b1 = (n * suma_de_producto - suma_de_x * suma_de_y) / (n * suma_de_cuadrado_x - suma_de_x * suma_de_x)
    b0 = (suma_de_y - b1 * suma_de_x) / n
    
    return b0, b1

def train_model(x, y):
    clf = LinearRegression()
    clf.fit(x, y)
    return clf
    
3. Archivo main.py

import matplotlib.pyplot as plt
from data_loading import load_data
from data_processing import process_data, calculate_manual_coefficients, train_model

# Cargar los datos
recuperado = load_data("/content/toneladas_recuperadas_2022-1_ok.xlsx")

# Procesar los datos
x, y = process_data(recuperado)

# Visualización inicial de los datos
plt.scatter(x, y)
plt.xlabel("Centro de Reciclaje de la Ciudad + Centros de Compostaje")
plt.ylabel("Centros Verdes de la Ciudad")
plt.grid()
plt.show()

# Convertir el DataFrame a una matriz numpy
matriz = recuperado.to_numpy()

# Calcular los coeficientes manualmente
b0, b1 = calculate_manual_coefficients(matriz)
print("Coeficiente b0 (manual):", b0)
print("Coeficiente b1 (manual):", b1)

# Entrenar el modelo de regresión lineal
clf = train_model(x, y)
print("Coeficiente (sklearn):", clf.coef_)
print("Intercepto (sklearn):", clf.intercept_)

# Predicción de un nuevo valor
prediccion = clf.predict([[62554.42]])
print("Predicción para 62554.42:", prediccion)

# Visualización de la regresión lineal ajustada
plt.scatter(x, y)
plt.plot(x, clf.predict(x), color='red')
plt.title("Regresión Lineal")
plt.xlabel("Centro de Reciclaje de la Ciudad + Centros de Compostaje")
plt.ylabel("Centros Verdes de la Ciudad")
plt.legend(["Predicciones", "Datos"])
plt.grid()
plt.show()

# Ecuación final de la regresión
print(f"Ecuación de la regresión: y = {clf.intercept_[0]} + {clf.coef_[0][0]}x")




