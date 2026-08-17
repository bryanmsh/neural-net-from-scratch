import network as nn
import data_loader as dl
import numpy as np

if __name__ == "__main__":
    training_labels = dl.load_labels("./MNIST/train-labels-idx1-ubyte")
    training_images = dl.load_img("./MNIST/train-images-idx3-ubyte")

    training_images = training_images / 255.0
    training_labels = dl.one_hot(training_labels)

    # test_labels = dl.load_labels("./MNIST/t10k-labels-idx1-ubyte")
    # test_images = dl.load_img("./MNIST/t10k-images-idx3-ubyte")


    network = nn.NeuralNetwork([784,16,16,10]) # create neural net
    output = network.forward(training_images[0]) # forward this row (this image)

    print(output)
    print(training_labels[0]) 


    for image, label in zip(training_images, training_labels): 
        # iterate through image and label at the same time, so the label is attached to the image
        output = network.forward(image)
        print(label, output)
