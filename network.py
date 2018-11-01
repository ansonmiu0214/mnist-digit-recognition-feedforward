import numpy as np

class Network:

  def __init__(self, sizes):
    """
    Initialises a neural network with layers defined by @param sizes.
    For example, Network([784, 20, 10]) defines a 3-layer neural network
    with 784/20/10 neurons in the 1st/2nd/3rd layers respectively.
    """
    self.num_layers = len(sizes)
    self.sizes = sizes

    # biases[i] := column vector of biases for neurons of layer[i]
    # No bias for 1st layer, so only considering sizes[1:]
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

    # weights[i] := matrix of weights connecting layer[i] to layer[i+1]
    # a neurons at layer[i] and b neurons at layer[i+1] require bxa transition matrix
    self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


  def feedforward(self, a):
    """
    Return the output of the network for input @param a.
    """
    for bias, weight in zip(self.biases, self.weights):
      a = sigmoid(np.dot(weight, a) + bias)
    return a

  
  



### Utility functions
def sigmoid(z):
  return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
  # Derivative of the sigmoid function
  sigmoid_val = sigmoid(z)
  return sigmoid_val / (1 - sigmoid_val)
  