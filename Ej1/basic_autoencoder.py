from Ej1.data.font import create_alphabet
from Ej1.multi_layer_perceptron import MultiLayerPerceptron
import math

class BasicAutoencoder:

    def __init__(self, alphabet, hidden_layers=1.0):
        self.alphabet = alphabet
        self.input_layer_size = len(alphabet[0]) # del tamaño de las entradas del "alfabeto"
        self.hidden_layers = hidden_layers
        self.total_layers = int(self.hidden_layers) + 2 # input + hidden + output
        self.build_network()


    """
        Network: cada nodo es un [layer, value], cada layer tiene la mitad de nodos que la capa anterior,
                 hasta el layer latente y ahi en adelante todas tienen el doble que la anterior hasta llegar
                 a la capa de salida con la misma cantidad de nodos que la capa de entrada.
                 Tengo 1 network => n layers => m nodos ==> [[[]]] triple lista
    """
    def build_network(self):
        nodes_per_layer = [0.0 for i in range(self.total_layers)]
        initial_count = self.input_layer_size
        for i in range(math.ceil(self.total_layers / 2)):
            amount = math.floor(initial_count / (2 ** i))
            nodes_per_layer[i] = amount
            nodes_per_layer[self.total_layers - 1 - i] = amount
        self.network = []
        for i in range(self.total_layers):
            self.network.append([[i, 0.0] for j in range(nodes_per_layer[i])])
        self.initialize_weights()
        print(self.network)
        
    def initialize_weights(self):
        self.weights = []