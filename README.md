**neural-net-from-scratch**

<img width="3150" height="1450" alt="2026-09-06 22-47" src="https://github.com/user-attachments/assets/254ec4ec-bff8-44ab-9171-28805fe59c8b" />

I challenged myself to code a basic neural network (multilayer perceptron) from scratch without the use of ml libraries to understand perceptrons and backpropagation from the ground up. This network uses the sigmoid activation function on each input and calculates cost using a mean squared function.

This uses NumPy for calculations and Matplotlib for a demo of the trained network.

The network currently predicts testing items from the MNIST dataset at ~90% accuracy. I have plans to implement a ReLu activation function instead of sigmoid and to use a cross entropy-loss function instead later on with softmax.

run main.py to test out the network

<img width="1270" height="1054" alt="modeldemo" src="https://github.com/user-attachments/assets/797e9e58-c12c-414e-93e2-157379d10d1a" />


run train.py to test out the training (on pre-existing parameters in params.npz)
