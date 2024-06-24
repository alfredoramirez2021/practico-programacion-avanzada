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
