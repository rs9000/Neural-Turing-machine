import torch
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys

class Controller(nn.Module):
	def __init__(self, num_inputs, num_outputs, num_hiddens):
		super(Controller, self).__init__()

		print("--- Initialize Controller")
		self.fc1 = nn.Linear(num_inputs, num_hiddens)
		self.fc2 = nn.Linear(num_hiddens, num_outputs)
		self.reset_parameters()

	def reset_parameters(self):
		# Initialize the linear layers
		nn.init.xavier_uniform_(self.fc1.weight, gain=1.4)
		nn.init.normal_(self.fc1.bias, std=0.01)

		nn.init.xavier_uniform_(self.fc2.weight, gain=1.4)
		nn.init.normal_(self.fc2.bias, std=0.01)


	def forward(self, x, last_read):
		x = Variable(torch.cat((x, torch.squeeze(last_read)), -1))
		x = F.sigmoid(self.fc1(x))
		x = F.sigmoid(self.fc2(x))
		return x
