import loader
from network import Network
from matplotlib import pyplot as plt
import numpy as np
from math import ceil

column_count = 4

def main(test_count=15):
  training_data, validation_data, test_data = loader.load_data_decorator()
  print("Data loaded.")

  training_data = list(training_data)

  layers = [784, 16, 10]
  network = Network(layers)
  print("Initialised network with layer structure {}".format(layers))

  network.SGD(training_data, 2, 10, 2.0, test_data)
  print("Training complete")

  count = 1
  rows = ceil(test_count / column_count)
  plt.tight_layout()
  for idx, (validation_input, validation_result) in enumerate(validation_data):
    output = network.feedforward(validation_input)
    print(output)

    idx = np.argmax(output)
    [output] = output[idx]

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
  main()