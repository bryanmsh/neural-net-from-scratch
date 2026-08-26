import numpy as np
import layer as l

class NeuralNetwork:

    def __init__(self, layer_sizes):
        self.layers = [l.Layer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]
        # create list of layers that take in input and output size, stopping at 2nd to last size
           
    def forward(self, a): # input activation
        for i in self.layers:
            a = i.forward(a) # loop through forwarding with respective input for each layer

        self.res = a
        return a

    def backward(self, output, actual):
        delta = 2*(output - actual) * l.sigmoid_deriv(self.layers[-1].ws) 
        # first delta is different since it comes from the cost, and must be calculated a bit differently
        W_next = None # output layer has no next layer
        for layer in reversed(self.layers):
            delta, W_next = layer.backward(delta, W_next)
            # as it loops through, each layer gives the info for next layer and stores the gradients of its own
            # no need to index next or before because the layers can find delta_next and w_next through calculating

if __name__ == "__main__":
    nn = NeuralNetwork([784,16,16,10]) # we can choose any number of parameters because of the way this is structured
    test = np.random.rand(784)
    nn.forward(test)
    print(nn.res)