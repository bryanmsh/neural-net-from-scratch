import numpy as np
import layer as l

class NeuralNetwork:

    def __init__(self, layer_sizes):
        self.layers = [l.Layer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]
        
        
    def forward(self, a): # input activation
        for i in self.layers:
            a = i.forward(a)
            
        nn.res = a
        return a


if __name__ == "__main__":
    nn = NeuralNetwork([784,16,16,10])
    test = np.random.rand(784)
    nn.forward(test)
    print(len(nn.res))