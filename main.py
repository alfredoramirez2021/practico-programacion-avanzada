from data_loader import DataLoader
from model import ConsumptionModel

def main():
    data_path = 'ruta/a/tu/archivo.csv'  # Especifica la ruta a tu archivo de datos

    # Cargar y preprocesar los datos
    loader = DataLoader(data_path)
    data = loader.load_data()
    X_train, X_test, y_train, y_test = loader.preprocess_data(data)

    # Crear, entrenar y evaluar el modelo
    model = ConsumptionModel()
    model.train(X_train, y_train)
    mse = model.evaluate(X_test, y_test)
    
    print(f"Mean Squared Error: {mse}")

    # Predecir sobre nuevos datos si es necesario
    # new_data = ...  # Obtén los nuevos datos de alguna manera
    # predictions = model.predict(new_data)
    # print(predictions)

if __name__ == "__main__":
    main()
