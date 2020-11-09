from Ej1.data.font import create_alphabet
from Ej1.multi_layer_perceptron import MultiLayerPerceptron
import math
import random
import numpy as np
import matplotlib.pyplot as plt

class BasicAutoencoder:

    def __init__(self, alphabet, epochs=1):
        self.alphabet = alphabet
        self.input_layer_size = len(alphabet[0]) # del tamaño de las entradas del "alfabeto"
        self.epochs = epochs
        self.alpha = 0.01
        self.beta = 0.5
        self.V = []                     # Valor de los nodos [capa, índice]
        self.W = []                     # Pesos [capa destino, nodo dest, nodo origen]
        self.d = []                     # Error
        self.total_layers = 0
        self.nodes_per_layer = []
        self.build_network()

    """
        Network: cada nodo es un [layer, value], cada layer tiene la mitad de nodos que la capa anterior,
                 hasta el layer latente y ahi en adelante todas tienen el doble que la anterior hasta llegar
                 a la capa de salida con la misma cantidad de nodos que la capa de entrada.
                 Tengo 1 network => n layers => m nodos ==> [[[]]] triple lista
    """
    def build_network(self):
        division_factor = 2
        initial_count = self.input_layer_size
        node_count = initial_count

        while node_count > 2:
            self.total_layers += 2
            node_count = math.floor(node_count / division_factor)
        self.total_layers += 1
        self.nodes_per_layer = [0 for i in range(self.total_layers)]
        node_count = initial_count
        for i in range(math.floor(self.total_layers/2)):
            self.nodes_per_layer[i] = node_count + 1                            # +1 por el bias
            self.nodes_per_layer[self.total_layers - 1 - i] = node_count + 1    # +1 por el bias
            node_count = math.floor(node_count / division_factor)
        self.nodes_per_layer[math.floor(self.total_layers/2)] = 3               # Dos de valor, uno para el bias
        self.nodes_per_layer[-1] -= 1                                           # Capa de salida no lleva bias
        for i in range(self.total_layers):
            print("Capa", i, ": ", self.nodes_per_layer[i], "Nodos")
            self.V.append([0.0 for j in range(self.nodes_per_layer[i])])
        self.initialize_weights()


    """
        Weights: array de tamaño 1 menos que la cantidad de layers, y en cada posición tiene un array con la cantidad de
                 valores de la capa siguiente, osea, weights[0] tiene una cantidad de pesos igual a la capa[1] * capa[0] de la red.
                 Esos valores son aleatorios con distribución uniforme entre 0 y 1
    """
    def initialize_weights(self):
        self.W = []
        self.W.append(np.random.rand(0,0))
        self.M = self.total_layers - 1
        for i in range(0, self.M):
            self.V[i][0] = 1                # Bias para cada capa
        for layer in range(self.M):
            w = np.random.rand(self.nodes_per_layer[layer+1], self.nodes_per_layer[layer]) - 0.5
            self.W.append(w)

    """
    def g(self, x):
        # ReLu
        return math.log(1.0 + math.exp(x))

    def g_derivative(self, x):
        return 1.0 / (1.0 + math.exp(x))
    """

    def g(self, x):
        return np.tanh(self.beta * x)

    def g_derivative(self, x):
        cosh2 = (np.cosh(self.beta*x)) ** 2
        return self.beta / cosh2


    def get_sum(self, neurons, weights, amount, index):
        sum_ = 0.0
        curr_neuron_index = 0
        for i in range(len(neurons)):
            sum_ += neurons[i] * weights[index * amount + i]
        return sum_

    def h(self, m, i, amount_of_nodes, W, V):
        hmi = 0
        for j in range(0, amount_of_nodes):
            hmi += W[m][i][j] * V[m-1][j]
        return hmi

    def train(self):
        #self.initialize_weights()
        data = self.alphabet
        self.M = self.total_layers - 1
        for layer in range(self.total_layers):
            self.d.append(np.zeros(self.nodes_per_layer[layer]))
        for i in range(1, self.M):
            self.V[i][0] = 1                 # Bias para cada capa

        for epoch in range(self.epochs):
            np.random.shuffle(data)
            for mu in range(len(data)):
                # Paso 2 (V0 tiene los ejemplos iniciales)
                self.V[0][0] = 1.0  # bias
                for k in range(len(data[0])):
                    self.V[0][k+1] = data[mu][k]

                # Paso 3 (Vi tiene los resultados de cada perceptron en la capa m. Salteo el nodo bias)
                for m in range(1, self.M):
                    for i in range(1, self.nodes_per_layer[m]):
                        hmi = self.h(m, i, self.nodes_per_layer[m-1], self.W, self.V)
                        self.V[m][i] = self.g(hmi)

                # Paso 3B (En la ultima capa no hay nodo bias)
                for i in range(self.nodes_per_layer[self.M]):
                    hmi = self.h(self.M, i, self.nodes_per_layer[self.M-1], self.W, self.V)
                    self.V[self.M][i] = self.g(hmi)

                # Paso 4 (Calculo error para capa de salida M)
                for i in range(0, self.nodes_per_layer[self.M]):
                    hMi = self.h(self.M, i, self.nodes_per_layer[self.M-1], self.W, self.V)
                    self.d[self.M][i] = self.g_derivative(hMi)*(data[mu][i] - self.V[self.M][i])

                # Paso 5 (Retropropagar error)
                for m in range(self.M, 1, -1):                                           # m es la capa superior
                    for i in range(0, self.nodes_per_layer[m-1]):
                        hprevmi = self.h(m-1, i, self.nodes_per_layer[m-2], self.W, self.V)
                        error_sum = 0
                        for j in range(0, self.nodes_per_layer[m]):                        # Por cada nodo en la capa superior
                            error_sum += self.W[m][j][i] * self.d[m][j]                    # sumo la rama de aca hasta arriba y multiplico por el error
                        self.d[m-1][i] = self.g_derivative(hprevmi) * error_sum

                # Paso 6 (Actualizar pesos)
                for m in range(1, self.M+1):
                    for i in range(self.nodes_per_layer[m]):
                        for j in range(self.nodes_per_layer[m-1]):
                            delta = self.alpha * self.d[m][i] * self.V[m-1][j]
                            self.W[m][i][j] = self.W[m][i][j] + delta
        # Show what the error was for the last letter used to train:
        """
        print("Input: ", self.V[0][1:])
        print("Output: ", self.V[-1])
        print("\n")
        print("Error:")
        print(abs(np.array(self.V[0][1:]) - np.array(self.V[-1])))
        """

    def test(self, test_data):
        print("\n\nExpectation / Reality")
        self.M = self.total_layers - 1
        for input in test_data:
            print("\n\n")
            for k in range(len(input)):
                self.V[0][k+1] = input[k]
            for m in range(1, self.M):
                for i in range(1, self.nodes_per_layer[m]):
                    hmi = self.h(m, i, self.nodes_per_layer[m-1], self.W, self.V)
                    self.V[m][i] = self.g(hmi)
            for i in range(0, self.nodes_per_layer[self.M]):
                hMi = self.h(self.M, i, self.nodes_per_layer[self.M-1], self.W, self.V)
                self.V[self.M][i] = self.g(hMi)
            perceptron_output = self.V[self.M]
            for bit in range(len(perceptron_output)):
                if(perceptron_output[bit] > 0): perceptron_output[bit] = 1
                else: perceptron_output[bit] = -1
            #Print the original letter
            for j in range(7):
                for i in range(5):
                    if(input[i+j*5] > 0): print("X", end = "")
                    else: print(".", end = "")
                print("\t", end="")
                for i in range(5):
                    if(perceptron_output[i+j*5] > 0): print("X", end = "")
                    else: print(".", end = "")
                print("")

    def graph(self, data, data_labels):
        self.M = self.total_layers - 1
        index = 0
        for input in data:
            for k in range(len(input)):
                self.V[0][k+1] = input[k]
            for m in range(1, self.M):
                for i in range(1, self.nodes_per_layer[m]):
                    hmi = self.h(m, i, self.nodes_per_layer[m-1], self.W, self.V)
                    self.V[m][i] = self.g(hmi)
            for i in range(0, self.nodes_per_layer[self.M]):
                hMi = self.h(self.M, i, self.nodes_per_layer[self.M-1], self.W, self.V)
                self.V[self.M][i] = self.g(hMi)
            perceptron_output = self.V[self.M]
            x = self.V[math.floor(self.total_layers/2)][1]
            y = self.V[math.floor(self.total_layers/2)][2]
            plt.scatter(x, y)
            plt.annotate(data_labels[index], xy=(x,y), textcoords='data')
            index += 1
        plt.grid()
        plt.show()
