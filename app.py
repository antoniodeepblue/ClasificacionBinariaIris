# Importar librerías
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib

# Cargar el Dataset Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Convertimos a DataFrame para una manipulación más fácil
iris_df = pd.DataFrame(X, columns=iris.feature_names)
iris_df['target'] = y

# Convertir a clasificación binaria: 1 para setosa, 0 para el resto
iris_df['binary_target'] = iris_df['target'].apply(lambda x: 1 if x == 0 else 0)
y_binary = iris_df['binary_target'].values

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

# Crear y entrenar el modelo
model = LogisticRegression()
model2= model.fit(X_train, y_train)

# Definir una función para la predicción
def modelo(sepal_length, sepal_width, petal_length, petal_width):
    species = ['Iris-Versicolour-Virginica','Setosa']
    i = model2.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    return species[i]

# Definir la interfaz Gradio
iface = gr.Interface(
    fn=modelo,
    inputs=[
        gr.Slider(minimum=0, maximum=10, label="Sepal Length"),
        gr.Slider(minimum=0, maximum=10, label="Sepal Width"),
        gr.Slider(minimum=0, maximum=10, label="Petal Length"),
        gr.Slider(minimum=0, maximum=10, label="Petal Width"),
    ],
    outputs=gr.Textbox(label='Specie'),
    live=True,
    title='Clasificacion de especies de iris, con Regresión',
    description='Este modelo está desarrollado para la clasificación Multiclase de flores de la especie Iris.',
    article= 'Autor: <a href=\"https://huggingface.co/Antonio49\">Antonio Fernández</a> de <a href=\"https://saturdays.ai/\">SaturdaysAI</a>. Formación: <a href=\"https://cursos.saturdays.ai/courses/\">Cursos Online AI</a> Aplicación desarrollada con fines docentes',
    theme='peach',
    examples = [[5,7,0,0],
            [0,1,2,8]]
)

# Ejecutar la interfaz
iface.launch()