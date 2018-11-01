import gzip
import pickle
import numpy as np

def load_data(filename='mnist.pkl.gz'):
  """
  Returns the MNIST data as (training_data, validation_dat, test_data)
  Each tuple entry is formatted as a tuple of (input, output).
  For the case of MNIST, the input is an array of digit images: each image is a 28*28 array.
  The output is an array of digits for the corresponding images
  """
  f = gzip.open(filename, 'rb')
  training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
  f.close()
  return training_data, validation_data, test_data

def load_data_decorator():
  """
  Decorate the output from `load_data` to be compatible with the network.
  Input images are converted from a 28*28 matrix to a 784-dimensional vector.
  Output digits are vectorised to reflect the final layer of the network.
  """
  (tr_in, tr_res), (va_in, va_res), (test_in, test_res) = load_data()

  training_inputs = [np.reshape(x, (784, 1)) for x in tr_in]
  training_results = [vectorise_result(y) for y in tr_res]
  training_data = zip(training_inputs, training_results)

  validation_inputs = [np.reshape(x, (784, 1)) for x in va_in]
  validation_data = zip(validation_inputs, va_res)

  test_inputs = [np.reshape(x, (784, 1)) for x in test_in]
  test_data = zip(test_inputs, test_res)

  return training_data, validation_data, test_data
  

def vectorise_result(i):
  """
  Return a 10-dimensional unit vector with 1.0 in the ith position.
  """
  output = np.zeros((10, 1))
  output[i] = 1.0
  return output