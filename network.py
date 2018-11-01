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
    Train the neural network using mini-batched SGD to update the network weights and biases.
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
    Updates the weights and biases through a single iteration of gradient descent
    using the training data in @param mini_batch and learning rate @param eta.
    """

    # Initialise gradient vectors
    nabla_b = [np.zeros(bias.shape) for bias in self.biases]
    nabla_w = [np.zeros(weight.shape) for weight in self.weights]

    for x, y in mini_batch:
      # Compute gradients of cost funtion using backpropagation
      delta_nabla_b, delta_nabla_w = self.backprop(x, y)

      # Update gradient accordingly
      nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
      nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

    # Apply the gradient changes to the network weights and biases
    m = len(mini_batch)
    self.weights = [w - (eta / m) * nw for w, nw in zip(self.weights, nabla_w)]
    self.biases = [b - (eta / m) * nb for b, nb in zip(self.biases, nabla_b)]

  
  def backprop(self, x, y):
    """
    Returns a tuple (nabla_b, nabla_w) representing the gradient for the cost function.
    """

    # Initialise gradient vectors
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]

    # Feedforward
    # The initial activation is the network input
    activation = x
    activations = [x]
    zs = []
    for bias, weight in zip(self.biases, self.weights):
      z = np.dot(weight, activation) + bias
      zs.append(z)

      # Compute activation for next layer using sigmoid
      activation = sigmoid(z)
      activations.append(activation)

    # Backward pass
    # Compute error (delta) vector and backpropagate the erpr
    delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())

    for l in range(2, self.num_layers):
      # Using the `negative indices` in Python
      # zs[-2] := 2nd-last output
      z = zs[-l]
      sp = sigmoid_prime(z)
      delta = np.dot(self.weights[-l+1].transpose(), delta) * sp

      # Update gradients
      nabla_b[-l] = delta
      nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

    return nabla_b, nabla_w

  
  def evaluate(self, test_data):
    """
    Return the number of test inputs for which tne neural network outputs the correct result.
    Neural network output inferred from the index (0-9) where the final layer has the highest activation.
    """
    test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
    return sum([int(x == y) for x, y in test_results])

  
  def cost_derivative(self, output_activations, y):
    """
    Return the vector of partial derivatives dC_x / d_a for the output activations.
    """
    return output_activations - y
      


### Utility functions
def sigmoid(z):
  return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
  # Derivative of the sigmoid function
  sigmoid_val = sigmoid(z)
  return sigmoid_val * (1 - sigmoid_val)
  