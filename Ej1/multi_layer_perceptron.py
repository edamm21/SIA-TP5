from matplotlib import pyplot as plt 
import numpy as np
import random
from datetime import datetime

class MultiLayerPerceptron:

    def __init__(self, alpha=0.01, beta=2.0, iterations=100, hidden_layers=1, error_tolerance=0.01, adaptive=False, classification_margin=0.99, nodes_per_layer=10, training_set_size=7):
        self.alpha = alpha
        self.initial_alpha = alpha
        self.beta = beta
        self.classification_margin = classification_margin
        self.iterations = iterations
        self.adaptive = adaptive
        self.hidden_layers = hidden_layers
        self.total_layers = hidden_layers + 2 # input + hidden + output
        self.layer_real_results = [None] * self.total_layers
        self.layer_activations = [None] * self.total_layers
        self.layer_weights = [[None] * self.total_layers]
        self.deltas_per_layer = [None] * self.total_layers
        self.error_tolerance = error_tolerance
        self.nodes_per_layer = nodes_per_layer
        self.training_set_size = training_set_size

    def adjust_learning_rate(self, errors_so_far):
        if(len(errors_so_far) > 10):
            last_10_errors = errors_so_far[-10:]
            booleans = []
            for i in range(len(last_10_errors) - 1):
                booleans.append(last_10_errors[i] > last_10_errors[i + 1])
            if all(booleans):
                self.alpha += 0.001
            else:
                self.alpha -= 0.01 * self.alpha

    def g(self, x):
        return np.tanh(self.beta * x)
 
    def g_derivative(self, x):
        cosh2 = (np.cosh(self.beta*x)) ** 2
        return self.beta / cosh2

    def h(self, m, i, amount_of_nodes, W, V):
        hmi = 0
        for j in range(0, amount_of_nodes):
            hmi += W[m,i,j] * V[m-1][j]
        return hmi

    def algorithm(self, problem):
        #                           bias    x     y    out
        """ if problem == "XOR": data = [[1.0,  1.0,  1.0, -1.0],
                                     [1.0, -1.0,  1.0,  1.0],
                                     [1.0,  1.0, -1.0,  1.0],
                                     [1.0, -1.0, -1.0, -1.0]]
        if problem == "EVEN":
            r = Reader('Ej3')
            data = r.readFile(size=self.training_set_size) """
                                
        self.M = self.total_layers - 1                                  # M sera el indice de la capa superior
        self.nodes_per_layer = max(self.nodes_per_layer, len(data[0]) - 1)                 # Cuantos nodos hay en las capas ocultas (incluye el del bias)
        self.exit_nodes = 1                                             # Cuantos nodos hay en la capa superior
        self.V = np.zeros((self.M + 1, self.nodes_per_layer))           # [capa, j]
        for i in range(1, self.M):
            self.V[i][0] = 1                                                                # Bias para cada capa
        self.W = np.random.rand(self.M+1, self.nodes_per_layer, self.nodes_per_layer)-0.5   # [capa destino, dest, origen]
        w = np.random.rand(self.nodes_per_layer, len(data[0]) - 1)-0.5                      # [dest, origen]
        self.W[1,:,:] = np.zeros((self.nodes_per_layer, self.nodes_per_layer))
        self.d = np.zeros((self.M+1, self.nodes_per_layer))
        for orig in range(len(data[0])-1):
            for dest in range(self.nodes_per_layer):
                self.W[1,dest,orig] = w[dest,orig]
        
        error_min = len(data)*2
        positivity_margin = self.classification_margin
        total_error = 1
        error_per_epoch = []
        worst_error_per_epoch = []
        accuracy = []
        plotter = Plotter()
        if problem == "EVEN":
            test_data = r.readFile(size=self.training_set_size, test=True)
        else:
            test_data = [[1.0,  1.0,  1.0, -1.0],
                         [1.0, -1.0,  1.0,  1.0],
                         [1.0,  1.0, -1.0,  1.0],
                         [1.0, -1.0, -1.0, -1.0]]
        test_error_per_epoch = []
        test_worst_error_per_epoch = []
        test_accuracy = []

        # LOOP
        for epoch in range(1, self.iterations):
            total_error = 0
            worst_error_this_epoch = 0
            positives = 0
            negatives = 0
            # Randomize W every once in a while
            if (epoch % 100000 == 99999):
                self.W = np.random.rand(self.M+1, self.nodes_per_layer, self.nodes_per_layer)-0.5   # [capa destino, dest, origen]
                w = np.zeros((nodes_per_layer, len(data[0]) - 1))                                   # [dest, origen]
                self.W[1,:,:] = np.zeros((self.nodes_per_layer, self.nodes_per_layer))
                for orig in range(len(data[0])-1):
                    for dest in range(nodes_per_layer):
                        self.W[1,dest,orig] = w[dest,orig]
            np.random.shuffle(data)
            for mu in range(len(data)):
                # Paso 2 (V0 tiene los ejemplos iniciales)
                for k in range(len(data[0])-1):
                    self.V[0][k] = data[mu][k]
                
                # Paso 3A (Vi tiene los resultados de cada perceptron en la capa m)
                for m in range(1, self.M):
                    for i in range(1, self.nodes_per_layer):
                        hmi = self.h(m, i, self.nodes_per_layer, self.W, self.V)
                        self.V[m][i] = self.g(hmi)

                # Paso 3B (En la ultima capa habra exit_nodes en vez de nodes_per_layer)
                for i in range(0, self.exit_nodes):
                    hMi = self.h(self.M, i, self.nodes_per_layer, self.W, self.V)
                    self.V[self.M][i] = self.g(hMi)
                upper_limit = data[mu][-1] + positivity_margin
                lower_limit = data[mu][-1] - positivity_margin
                if self.V[self.M][i] >= lower_limit and self.V[self.M][i] <= upper_limit:
                    positives += 1
                else:
                    negatives += 1

                # Paso 4 (Calculo error para capa de salida M)
                for i in range(0, self.exit_nodes):
                    hMi = self.h(self.M, i, self.nodes_per_layer, self.W, self.V)
                    if self.exit_nodes == 1:
                        self.d[self.M][i] = self.g_derivative(hMi)*(data[mu][-1] - self.V[self.M][i])
                    else:
                        self.d[self.M][i] = self.g_derivative(hMi)*(data[mu][-1][i] - self.V[self.M][i])

                # Paso 5 (Retropropagar error)
                for m in range(self.M, 1 ,-1):                                           # m es la capa superior
                    for j in range(0, self.nodes_per_layer):                             # Por cada j en el medio
                        hprevmi = self.h(m-1, j, self.nodes_per_layer, self.W, self.V)   # hj = hj del medio
                        error_sum = 0
                        for i in range(0, self.nodes_per_layer):                         # Por cada nodo en la capa superior
                            error_sum += self.W[m,i,j] * self.d[m][i]                    # sumo la rama de aca hasta arriba y multiplico por el error
                        self.d[m-1][j] = self.g_derivative(hprevmi) * error_sum

                # Paso 6 (Actualizar pesos)
                for m in range(1, self.M+1):
                    for i in range(self.nodes_per_layer):
                        for j in range(self.nodes_per_layer):
                            delta = self.alpha * self.d[m][i] * self.V[m-1][j]
                            self.W[m,i,j] = self.W[m,i,j] + delta

                # Paso 7 (Calcular error)
                for i in range(0, self.exit_nodes):
                    if abs(data[mu][-1] - self.V[self.M][i]) > worst_error_this_epoch:
                        worst_error_this_epoch = abs(data[mu][-1] - self.V[self.M][i])
                    if self.exit_nodes == 1:
                        total_error += abs(data[mu][-1] - self.V[self.M][i])
                    else:
                        total_error += abs(data[mu][-1][i] - self.V[self.M][i])
            error_per_epoch.append(total_error/len(data))
            worst_error_per_epoch.append(worst_error_this_epoch)
            accuracy.append(positives / (0.0 + positives + negatives))
            if self.adaptive and epoch % 10 == 0:
                self.adjust_learning_rate(error_per_epoch)
            if total_error < error_min:
                error_min = total_error
                self.w_min = self.W
            if total_error <= self.error_tolerance*len(data) or epoch == self.iterations-1:
                self.test_perceptron(test_data, self.w_min, epoch, test_error_per_epoch, test_worst_error_per_epoch, test_accuracy, positivity_margin, True)
                break
            else:
                self.test_perceptron(test_data, self.w_min, epoch, test_error_per_epoch, test_worst_error_per_epoch, test_accuracy, positivity_margin, False)
        plotter.create_plot_ej3(error_per_epoch, worst_error_per_epoch, test_error_per_epoch, test_worst_error_per_epoch)
        plotter.create_plot_ej3_accuracy(accuracy, test_accuracy)
        return

    def test_perceptron(self, test_data, weights, epoch, test_error, test_worst_error, test_accuracy, positivity_margin, printing):
        element_count = 0
        if printing:
            print("Testing perceptron for epoch %d..." %(epoch+1))
            print('+-------------------+-------------------+')
            print('|   Desired output  |   Perceptron out  |')
            print('+-------------------+-------------------+')
        total_error = 0
        worst_error = 0
        positives = 0
        negatives = 0
        W = weights
        for row in test_data:
            for k in range(len(row)-1):
                self.V[0][k] = row[k]
            for m in range(1, self.M):
                for i in range(1, self.nodes_per_layer):
                    hmi = self.h(m, i, self.nodes_per_layer, W, self.V)
                    self.V[m][i] = self.g(hmi)
            for i in range(0, self.exit_nodes):
                hMi = self.h(self.M, i, self.nodes_per_layer, W, self.V)
                self.V[self.M][i] = self.g(hMi)
            perceptron_output = self.V[self.M][0]
            element_count += 1
            if printing:
                print('       {}\t    |  {}'.format(row[-1], perceptron_output))
            upper_limit = row[-1] + positivity_margin
            lower_limit = row[-1] - positivity_margin
            if perceptron_output >= lower_limit and perceptron_output <= upper_limit:
                positives += 1
            else:
                negatives += 1
            if abs(perceptron_output- row[-1]) > worst_error:
                worst_error = abs(perceptron_output- row[-1])
            total_error += abs(perceptron_output- row[-1])
        test_error.append(total_error/len(test_data))
        test_worst_error.append(worst_error)
        test_accuracy.append(positives / (0.0 + positives + negatives))
        if printing:
            print('Analysis finished for epoch %d' %(epoch+1))
            print('Initial learning rate: {}'.format(self.initial_alpha))
            print('End learning rate: {}'.format(self.alpha))
            print('+-------------------+-------------------+')
            print('Test finished')