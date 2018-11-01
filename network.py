import numpy as np
import random

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

  
  def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    """

    """
    
    training_data = list(training_data)
    n = len(training_data)
  
    if test_data:
      test_data = list(test_data)
      n_test = len(test_data)

    for idx in range(epochs):
      # Shuffle and slice training data by the mini_batch_size
      random.shuffle(training_data)
      mini_batches = [training_data[j:j+mini_batch_size] for j in range(0, n, mini_batch_size)]

      # Apply SGD using backpropagation on each minibatch
      for mini_batch in mini_batches:
        self.update_mini_batch(mini_batch, eta)
      
      # Print evaluation result if applicable
      if test_data:
        print("Epoch {}: {} / {}".format(idx, self.evaluate(test_data), n_test))
      else:
        print("Epoch {} complete".format(idx))
    

    def update_mini_batch(self, mini_batch, eta):
      """

      """
      pass
    

    def backprop(self, x, y):
      """

      """
      pass
    
    def evaluate(self, test_data):
      """

      """
      pass

      


### Utility functions
def sigmoid(z):
  return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
  # Derivative of the sigmoid function
  sigmoid_val = sigmoid(z)
  return sigmoid_val / (1 - sigmoid_val)
  