import numpy as np
import layer as l

class NeuralNetwork:

    def __init__(self, layer_sizes):
        self.layers = [l.Layer(i, i+1) for i in range(len(layer_sizes)-1)]
        
        
    def forward():
        pass

if __name__ == "__main__":
    NeuralNetwork([728,16,16,10])