import numpy as np

def sigmoid(num):
    return 1 /(1+np.exp(-num)) # activation function to smoothen inputs

class Layer:

    def __init__(self, input_size, output_size):
        self.w = np.random.randn(output_size, input_size) * 0.01
         # initialize weights with random numbers, 
         # bell-shaped, with some negatives - use randn
         # scale down by .01 so weighted sum is not extremely large
        self.b = np.zeros(output_size) # initialize all biases as zero

    def forward(self, a): # input activation
        self.ws = self.w @ a + self.b # multiply inputs and weights for each neuron and add bias to get weighted sum
        # print(self.ws)
        self.output = sigmoid(self.ws) # apply sigmoid activation function to get output
        return self.output


if __name__ == "__main__":
    layer = Layer(784,16)
    input = np.random.rand(784)
    layer.forward(input)

   
