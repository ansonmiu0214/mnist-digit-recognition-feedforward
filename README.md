# mnist-digit-recognition-feedforward
Implementing handwritten digit recognition on the MNIST dataset using a feedfoward neural network.

## Getting started
Tested on Python 3.7.

### Installation
1. Create a virtual environment `venv` in the current working directory using `python3 -m venv venv`.
2. Activate the virtual environment through `source venv/bin/activate`.
3. Install the dependencies from `requirements.txt` using `pip install -r requirements.txt`.

### Usage
```
python3 predict.py -e 30 -l 32 16 -t 20
```
Sets up a neural network with hidden layers of [32, 16].
Trains using 30 epochs.
Runs through 20 validation cases after training.

## Acknowledgements
The eBook on [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html) helped guide me through building and understanding the concepts of how a neural network functions.