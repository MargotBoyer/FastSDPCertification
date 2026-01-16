import torch.nn as nn

def MNIST_MLP():
	model = nn.Sequential(
		nn.Flatten(),
		nn.Linear(784, 100),
		nn.ReLU(),
		nn.Linear(100, 100),
		nn.ReLU(),
		nn.Linear(100, 10)
	)
	return model