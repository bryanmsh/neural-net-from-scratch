import numpy as np

def save_params(network, path="params.npz"): # pass in a neural net (NeuralNetwork obj)
    params = {}
    for i, layer in enumerate(network.layers):
        params[f"w{i}"] = layer.W
        params[f"b{i}"] = layer.B
    np.savez(path,**params)
    # make a dictionary and pack params in, then save the entire array

def load_params(network, path="params.npz"):
    params = np.load(path) # load parameters file
    for i, layer in enumerate(network.layers):
        layer.W = params[f"w{i}"]
        layer.B = params[f"b{i}"]
    # opposite of save
        



if __name__ == "__main__":

    a = np.array([1, 0, 1, 1, 0])
    b = np.array([0, 1, 0, 0, 0])

    import network as nn
    # for i, layer in enumerate(b):
    #     print(i, layer)
    sig = nn.NeuralNetwork([784,16,16,10])
    save_params(sig)
    print()
    # print(np.load("sigma.npy"))