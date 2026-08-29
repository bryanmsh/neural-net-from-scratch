import network as nn
import data_loader as dl
import numpy as np
from params import save_params, load_params
import train 
import matplotlib.pyplot as plot

def check_accuracy():
    test_labels = dl.load_labels("./MNIST/t10k-labels-idx1-ubyte")
    test_images = dl.load_img("./MNIST/t10k-images-idx3-ubyte")

    network = nn.NeuralNetwork([784,16,16,10]) # create neural net
    load_params(network)

    count = len(test_labels)
    total = count

    for image, label in zip(test_images, test_labels): 
        # iterate through image and label at the same time, so the label is attached to the image
        output = network.forward(image)
        
        np.set_printoptions(suppress=True, precision=6)


        if np.argmax(output) != label:
            #print(np.argmax(output), label)
            
            # image = image.reshape(28, 28)
            # # reshape to the 28x28 image for visualization

            # figure, (ax1, ax2) = plot.subplots(1, 2)
            # # create both subplots 
            
            # ax1.imshow(image.reshape(28, 28), cmap="gray")
            # ax1.set_title(f"Image of a {label}")
            # ax1.axis("off")
            # # config image plot
            
            # ax2.bar(range(10), network.res)
            # ax2.set_xticks(range(10))
            # ax2.set_xlabel("Digit")
            # ax2.set_ylabel("Activation")
            # ax2.set_title("Network Output")
            # # config network visualization plot


            # plot.tight_layout()
            # plot.show()

            total -= 1

    print(f"Accuracy: {total/count}")

if __name__ == "__main__":

    check_accuracy()
    # exit()

    test_labels = dl.load_labels("./MNIST/t10k-labels-idx1-ubyte")
    test_images = dl.load_img("./MNIST/t10k-images-idx3-ubyte")

    test_images = test_images / 255.0
    test_labels = dl.one_hot(test_labels)


    network = nn.NeuralNetwork([784,16,16,10]) # create neural net
    load_params(network)
    
    #output = network.forward(training_images[0]) # forward this row (this image)
    # np.set_printoptions(suppress=True, precision=6)
    # print(output)
    # print(training_labels[1]) 

    for image, label in zip(test_images, test_labels): 
        # iterate through image and label at the same time, so the label is attached to the image
        output = network.forward(image)
        
        np.set_printoptions(suppress=True, precision=6)
        print(output)
        print(np.argmax(output))
        print(label) 


        image = image.reshape(28, 28)
        # reshape to the 28x28 image for visualization

        figure, (ax1, ax2) = plot.subplots(1, 2)
        # create both subplots 
        
        ax1.imshow(image.reshape(28, 28), cmap="gray")
        ax1.set_title(f"Image of a {np.argmax(label)}")
        ax1.axis("off")
        # config image plot
        
        ax2.bar(range(10), network.res)
        ax2.set_xticks(range(10))
        ax2.set_xlabel("Digit")
        ax2.set_ylabel("Activation")
        ax2.set_title("Network Output")
        # config network visualization plot


        plot.tight_layout()
        plot.show()

