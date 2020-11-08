from Ej1.data.font import create_alphabet
from Ej1.multi_layer_perceptron import MultiLayerPerceptron
import math
import random
import numpy as np

class BasicAutoencoder:

    def __init__(self, alphabet, epochs=1):
        self.alphabet = alphabet
        self.input_layer_size = len(alphabet[0]) # del tamaño de las entradas del "alfabeto"
        self.epochs = epochs
        self.build_network()

    """
        Network: cada nodo es un [layer, value], cada layer tiene la mitad de nodos que la capa anterior,
                 hasta el layer latente y ahi en adelante todas tienen el doble que la anterior hasta llegar
                 a la capa de salida con la misma cantidad de nodos que la capa de entrada.
                 Tengo 1 network => n layers => m nodos ==> [[[]]] triple lista
    """
    def build_network(self):
        initial_count = self.input_layer_size
        node_count = initial_count
        self.network = []
        self.total_layers = 0        
        while node_count > 2:
            self.total_layers += 2
            node_count = math.floor(node_count / 4)
        self.total_layers += 1
        self.nodes_per_layer = [0 for i in range(self.total_layers)]
        node_count = initial_count
        for i in range(math.floor(self.total_layers/2)):
            print(node_count)
            self.nodes_per_layer[i] = node_count + 1 # +1 por el bias 
            self.nodes_per_layer[self.total_layers - 1 - i] = node_count + 1 # +1 por el bias 
            node_count = math.floor(node_count / 4)
        self.nodes_per_layer[math.floor(self.total_layers/2)] = 2
        for i in range(self.total_layers):
            self.network.append([0.0 for j in range(self.nodes_per_layer[i])]) 
        self.initialize_weights()
        
    
    """
        Weights: array de tamaño 1 menos que la cantidad de layers, y en cada posición tiene un array con la cantidad de
                 valores de la capa siguiente, osea, weights[0] tiene una cantidad de pesos igual a la capa[1] * capa[0] de la red.
                 Esos valores son aleatorios con distribución uniforme entre 0 y 1
    """
    def initialize_weights(self):
        self.weights = [None for i in range(self.total_layers - 1)]
        for i in range(len(self.nodes_per_layer) - 1):
            self.weights[i] = [random.uniform(0, 1) for j in range(self.nodes_per_layer[i+1] * self.nodes_per_layer[i] + 1)] 

    def activate(self, x):
        # ReLu
        return math.log(1.0 + math.exp(x))
    
    def derivative(self, x):
        return 1.0 / (1.0 + math.exp(x))

    def get_sum(self, neurons, weights, amount, index):
        sum_ = 0.0
        curr_neuron_index = 0
        for i in range(len(neurons)):
            sum_ += neurons[i] * weights[index * amount + i]
        return sum_

    def train(self):
        data = self.alphabet
        for epoch in range(self.epochs):
            np.random.shuffle(data)
            for input_ in data:
                
                # get input data into network's first layer
                for m in range(len(input_)):
                    self.network[0][m] = input_[m]
                self.network[0][len(input_)] = 1.0 # bias
                
                # Feeding forward to network
                for layer in range(self.total_layers - 1): 
                    for neuron in range(self.nodes_per_layer[layer + 1]): 
                        sum_ = self.get_sum(self.network[layer], self.weights[layer], self.nodes_per_layer[layer], neuron)
                        self.network[layer + 1][neuron] = self.activate(sum_)
                
