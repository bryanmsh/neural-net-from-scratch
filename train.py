import numpy as np
import network as nn
import data_loader as dl


def cost(output, label): # cost for 1 output
    return np.sum((output-label)**2)

def avg_cost(img_set, label_set):
    total = 0
    for image, label in zip(img_set, label_set): 
        total += cost(network.forward(image), label) 
        # sum cost of each output
    total /= len(label_set) # divide for avg
    return (total)


if __name__ == "__main__":
    training_labels = dl.load_labels("./MNIST/train-labels-idx1-ubyte")
    training_images = dl.load_img("./MNIST/train-images-idx3-ubyte")

    training_images = training_images / 255.0
    training_labels = dl.one_hot(training_labels)

    # output = np.array([0,0,0])
    # label = np.array([0,1,0])
    # print((output-label)**2)
    network = nn.NeuralNetwork([784,16,16,10])
    print(cost(network.forward(training_images[0]), training_labels[0]))
    print(avg_cost(training_images, training_labels))

