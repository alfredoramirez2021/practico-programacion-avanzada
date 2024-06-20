
class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        # Carga los datos desde el archivo CSV
        data = pd.read_csv(self.data_path)
        return data

    def preprocess_data(self, data):
        # Supongamos que la columna 'consumo' es el objetivo
        X = data.drop('consumo', axis=1)
        y = data['consumo']

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Escalar los datos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test
