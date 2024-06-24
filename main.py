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
