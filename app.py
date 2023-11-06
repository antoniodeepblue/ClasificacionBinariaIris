# Importar librerías
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import gradio as gr

# Cargar el conjunto de datos Iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Construir el modelo
model = Sequential([
    Dense(20, activation='relu', input_dim=4),  # Capa oculta con 10 neuronas y función de activación ReLU
    Dense(10, activation='relu'),  # Capa oculta con 10 neuronas y función de activación ReLU
    Dense(3, activation='softmax')  # Capa de salida con 3 neuronas para las clases de iris
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# Definir una función para la predicción
def predict_iris_species(sepal_length, sepal_width, petal_length, petal_width):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    scaled_input_data = scaler.transform(input_data)
    prediction = model.predict(scaled_input_data)
    return iris.target_names[np.argmax(prediction)]

# Definir la interfaz Gradio
iface = gr.Interface(
    fn=predict_iris_species,
    inputs=[
        gr.Slider(minimum=0, maximum=10, label="Sepal Length"),
        gr.Slider(minimum=0, maximum=10, label="Sepal Width"),
        gr.Slider(minimum=0, maximum=10, label="Petal Length"),
        gr.Slider(minimum=0, maximum=10, label="Petal Width"),
    ],
    outputs=gr.Label(num_top_classes=3),
    live=True,
    title='Detector de especies de iris, en Red Neuronal',
    description='Este modelo está desarrollado para la clasificación Multiclase de flores de la especie Iris.',
    article='Antonio Fernández Salcedo. Practicas Saturdays.AI. Aplicación desarrollada con fines docentes',
    theme='peach'   
)

# Ejecutar la interfaz
iface.launch()