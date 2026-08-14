import numpy as np
import layer as l

class NeuralNetwork:

    def __init__(self, layer_sizes):
        self.layers = [l.Layer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]
        # create list of layers that take in input and output size, stopping at 2nd to last size
           
    def forward(self, a): # input activation
        for i in self.layers:
            a = i.forward(a) # loop through forwarding with respective input for each layer

        nn.res = a
        return a


if __name__ == "__main__":
    nn = NeuralNetwork([784,16,16,10])
    test = np.random.rand(784)
    nn.forward(test)
    print(len(nn.res))