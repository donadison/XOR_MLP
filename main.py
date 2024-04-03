import numpy as np
import matplotlib.pyplot as plt

# Definiowanie funkcji aktywacji (sigmoid).
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Pochodna funkcji aktywacji (do propagacji wstecznej)
def sigmoid_derivative(x):
    return x * (1 - x)

#próg dla błędu binarnego
threshold=0.5

# Funkcja progowa dla klasyfikacji binarnej z progiem 0.5
def step_function(x, threshold):
    return np.where(x >= threshold, 1, 0)

# Dane wejściowe
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Oczekiwane wyniki (XOR)
y = np.array([[0],
              [1],
              [1],
              [0]])

# Inicjalizacja warstw
input_size = 2
hidden_size = 2
output_size = 1

# Wagi dla warstwy ukrytej
weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))

# Wagi dla warstwy wyjściowej
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

# Inicjalizacja tablic do przechowywania historii
errormse_history = []
weights_input_hidden_history = []
weights_hidden_output_history = []
output_history = []
binary_error_history = []
incorrect_indices = []
iterational_output = []

# Uczenie
learning_rate = 1.4
epochs = 10000

# Pętla ucząca
for epoch in range(epochs):
    # Przód
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output = sigmoid(output_layer_input)
    output_history.append(output)

    # Obliczanie błędu MSE
    errormse = np.mean(np.square(y - output))
    errormse_history.append(errormse)

    # Propagacja wsteczna
    error = y - output
    d_output = error * sigmoid_derivative(output)
    error_hidden_layer = d_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Aktualizacja wag
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate

    # Zapisywanie wag do historii
    weights_input_hidden_history.append(weights_input_hidden.copy())
    weights_hidden_output_history.append(weights_hidden_output.copy())

    # Klasyfikacja binarna z progiem 0.5
    binary_output = step_function(output, threshold)

    #alternatywny sposób błędu klasyfikacji według Politechniki Poznańskiej
    binary_error = np.sum(np.abs(y - binary_output)) / len(y)
    binary_error_history.append(binary_error)


# Wyniki
print("Wagi dla warstwy ukrytej po uczeniu:")
print(weights_input_hidden)
print("\nWagi dla warstwy wyjściowej po uczeniu:")
print(weights_hidden_output)
print("\nWyjścia po uczeniu:")
print(output)
print("\nKlasyfikacja binarna po uczeniu:")
print(binary_output)
print("\nWartości oczekiwane:")
print(y)

# Wykres błędu MSE
plt.plot(errormse_history)
plt.title('Błąd MSE w kolejnych epokach')
plt.xlabel('Epoka')
plt.ylabel('Błąd MSE')
plt.show()

# Wykres błędu klasyfikacji binarnej
plt.plot(binary_error_history)
plt.title('Błąd klasyfikacji binarnej w kolejnych epokach')
plt.xlabel('Epoka')
plt.ylabel('Błąd klasyfikacji binarnej')
plt.show()

# Wykres klasyfikacji binarnej po uczeniu
plt.scatter(range(len(y)), y, color='blue', label='Wartości oczekiwane')
plt.scatter(range(len(output)), output, color='red', marker='x', label='Wartosci wyjsciowe')
plt.axhline(y=0.5, color='gray', linestyle='--', label='próg kwalifikacji')
plt.title('Klasyfikacja binarna po uczeniu')
plt.xlabel('Próbka')
plt.ylabel('Wartość')
plt.legend()
plt.show()

# Wykresy wag w obu warstwach
weights_input_hidden_history = np.array(weights_input_hidden_history)
weights_hidden_output_history = np.array(weights_hidden_output_history)

for i in range(hidden_size):
    plt.plot(weights_input_hidden_history[:, :, i])
    plt.title('Wagi warstwy ukrytej - Neuron ' + str(i+1))
    plt.xlabel('Epoka')
    plt.ylabel('Waga')
    plt.show()

for i in range(output_size):
    plt.plot(weights_hidden_output_history[:, :, i])
    plt.title('Wagi warstwy wyjściowej - Wyjście ' + str(i+1))
    plt.xlabel('Epoka')
    plt.ylabel('Waga')
    plt.show()