import numpy as np
import struct

def load_img(path):
    with open(path, 'rb') as f:
        identifier, img_count, rows, cols = struct.unpack('>IIII', f.read(16)) # data from first 16 bytes, decoded as 4 big-endian, unsigned 32-bit integers
        if identifier != 2051: # this happens when the first 4 bytes are not 2051, meaning it is not the collection of images.
            raise ValueError(f"Incorrect identifier: {identifier}. This is not the correct file!")

        data = f.read() # reads the rest of the file as is
        images = np.frombuffer(data, dtype=np.uint8) # convert raw data to numpy array, each byte as 1 unsigned 8-bit int (0-255)
        images = images.reshape(img_count, rows * cols) # reshape from 1D array to 2D, with one row per image
        return images

def load_labels(path):
    with open(path, 'rb') as f:
        identifier, label_count = struct.unpack('>II', f.read(8)) # data from first 8 bytes, decoded as 4 big-endian, unsigned 32-bit integers
        if identifier != 2049: # this happens when the first 4 bytes are not 2049, meaning it is not the collection of images.
            raise ValueError(f"Incorrect identifier: {identifier}. This is not the correct file!")

        data = f.read() # reads the rest of the file as is (values 0-9)
        labels = np.frombuffer(data, dtype=np.uint8) # convert raw data to numpy array, each byte as 1 unsigned 8-bit int 
        # no need to reshape, stay as flat 60,000 num (0-9)
        return labels

def one_hot(labels, class_count=10):
    encoded = np.zeros((labels.shape[0], class_count)) # array of zeros, 10 columns, one row per label
    encoded[np.arange(labels.shape[0]), labels] = 1
    # index to each label in each row and change to one
    # used later to compare NN output to actual (from label)
    # looks like [0, 0, 0, 0, 1, 0, 0, 0, 0] representing 5
    return encoded

training_labels = load_labels("./MNIST/train-labels-idx1-ubyte")
training_images = load_img("./MNIST/train-images-idx3-ubyte")

test_labels = load_labels("./MNIST/t10k-labels-idx1-ubyte")
test_images = load_img("./MNIST/t10k-images-idx3-ubyte")

if __name__ == "__main__":
    pass