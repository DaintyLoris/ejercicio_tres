import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cargar el conjunto de datos "concentlite.csv"
data = pd.read_csv("concentlite.csv")

# Separar las características (X) y las etiquetas (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Definir la arquitectura de la red neuronal
# Puedes personalizar la cantidad de capas y neuronas en cada capa aquí
input_size = X.shape[1]  # Tamaño de entrada (número de características)
hidden_layer_sizes = [5, 5]  # Número de neuronas en cada capa oculta
output_size = 1  # Tamaño de salida (en este caso, es una clasificación binaria)


# Función de activación (usaremos la función sigmoide)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivada de la función sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)


# Inicializar los pesos y sesgos de la red neuronal de manera aleatoria
weights = []
biases = []
layer_sizes = [input_size] + hidden_layer_sizes + [output_size]

for i in range(1, len(layer_sizes)):
    w = np.random.randn(layer_sizes[i - 1], layer_sizes[i])
    b = np.zeros((1, layer_sizes[i]))
    weights.append(w)
    biases.append(b)

# Hiperparámetros
learning_rate = 0.1
epochs = 1000

# Entrenamiento de la red neuronal
for epoch in range(epochs):
    # Forward propagation
    layer_outputs = []
    layer_inputs = [X]

    for i in range(len(layer_sizes) - 1):
        z = np.dot(layer_inputs[-1], weights[i]) + biases[i]
        a = sigmoid(z)
        layer_outputs.append(a)
        layer_inputs.append(a)

    # Calcular el error
    error = y.reshape(-1, 1) - layer_outputs[-1]

    # Backpropagation
    deltas = [error * sigmoid_derivative(layer_outputs[-1])]

    for i in range(len(layer_sizes) - 2, 0, -1):
        delta = np.dot(deltas[-1], weights[i].T) * sigmoid_derivative(layer_outputs[i])
        deltas.append(delta)

    # Revertir la lista de deltas para que coincida con la dirección de propagación
    deltas = deltas[::-1]

    # Actualizar pesos y sesgos
    for i in range(len(weights)):
        weights[i] += learning_rate * np.dot(layer_inputs[i].T, deltas[i])
        biases[i] += learning_rate * np.sum(deltas[i], axis=0)

    # Calcular la precisión en cada iteración (opcional)
    predictions = (layer_outputs[-1] > 0.5).astype(int)
    accuracy = np.mean(predictions == y.reshape(-1, 1))

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy}")

# Representar gráficamente el resultado de la clasificación
plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap=plt.cm.Spectral)
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.title("Clasificación con Perceptrón Multicapa (2 Neuronas en la Capa de Salida)")
plt.show()
