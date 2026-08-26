from params import load_params, save_params
import numpy as np
import network as nn
import data_loader as dl


def cost(output, label): # cost for 1 output
    return np.sum((output-label)**2)

def avg_cost(network, img_set, label_set):
    total = 0
    for image, label in zip(img_set, label_set): 
        total += cost(network.forward(image), label) 
        # sum cost of each output
    total /= len(label_set) # divide for avg
    return (total)

def train(network, img_set, label_set, learning_rate, num_epochs=7):
    load_params(network)
    for epoch in range(1, num_epochs+1):

        total_dcdB = [np.zeros_like(layer.B) for layer in network.layers]
        total_dcdW = [np.zeros_like(layer.W) for layer in network.layers]
        # initialize totals, format same as the arrays they sum

        for image, label in zip(img_set, label_set):
            output = network.forward(image)
            network.backward(output, label)

            for i, layer in enumerate(network.layers):
                total_dcdB[i] += layer.dcdB
                total_dcdW[i] += layer.dcdW
                # cycle through each layer in network and sum gradients
                
        count = len(img_set)
        for i, layer in enumerate(network.layers):
            # need to cycle through layers again
            layer.W -= learning_rate * (total_dcdW[i]/count)
            layer.B -= learning_rate * (total_dcdB[i]/count)
        
            
        print(f"Total cost for epoch {epoch}: {avg_cost(network, img_set, label_set)}")

    save_params(network)
        

if __name__ == "__main__":
    training_labels = dl.load_labels("./MNIST/train-labels-idx1-ubyte")
    training_images = dl.load_img("./MNIST/train-images-idx3-ubyte")

    training_images = training_images / 255.0
    training_labels = dl.one_hot(training_labels)

    # output = np.array([0,0,0])
    # label = np.array([0,1,0])
    # print((output-label)**2)
    network = nn.NeuralNetwork([784,16,16,10])
    load_params(network)
    # print(cost(network.forward(training_images[0]), training_labels[0]))
    # print(avg_cost(network, training_images, training_labels))

    train(network, training_images, training_labels, 1, 15)
