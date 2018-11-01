import loader
from network import Network
from matplotlib import pyplot as plt
import numpy as np
from math import ceil
from argparse import ArgumentParser

column_count = 4

def main(epoch, hidden_layers, test_count):
  training_data, validation_data, test_data = loader.load_data_decorator()
  print("Data loaded.")

  training_data = list(training_data)

  layers = [784] + hidden_layers + [10]
  network = Network(layers)
  print("Initialised network with layer structure {}".format(layers))

  mini_batch_size = 10
  learning_rate = 2.0
  network.SGD(training_data, epoch, mini_batch_size, learning_rate, test_data)
  print("Training complete")

  count = 1
  rows = ceil(test_count / column_count)
  plt.tight_layout()
  for idx, (validation_input, validation_result) in enumerate(validation_data):
    output = network.feedforward(validation_input)

    idx = np.argmax(output)
    [output] = output[idx]

    # Reformat image to 28x28 for plotting
    image = np.reshape(validation_input, (28, 28))

    plt.subplot(rows, column_count, count)
    plt.imshow(image)
    plt.title('P({}) = {} ({})'.format(idx, round(output, 3), validation_result))
    plt.xticks([])
    plt.yticks([])

    count += 1
    if count == test_count:
      break

  plt.show()


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument('-e', '--epoch', type=int, default=30)
  parser.add_argument('-l', '--layers', nargs='+', type=int, default=16)
  parser.add_argument('-t', '--tests', type=int, default=15)

  args = parser.parse_args()
  main(args.epoch, args.layers, args.tests)