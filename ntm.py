import torch
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys

from memory import ReadHead, WriteHead
from controller import Controller

class NTM(nn.Module):
	def __init__(self, M, N, num_inputs, num_outputs, controller_out_dim, controller_hid_dim, learning_rate):
		super(NTM, self).__init__()

		print("----------- Build Neural Turing machine -----------")
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		self.M = M
		self.N = N

		self.learning_rate = learning_rate

		self.controller = Controller(self.num_inputs+N, controller_out_dim, controller_hid_dim)
		self.read_head = ReadHead(self.M, self.N, controller_out_dim)
		self.write_head = WriteHead(self.M, self.N, controller_out_dim)

		self.memory = []
		self.last_read = []
		self.mem_weights_read = []
		self.mem_weights_write = []

		self.fc_out = nn.Linear(self.num_inputs+N, self.num_outputs)
		self._initalize_state()
		self.reset_parameters();

	def forward(self, X=None):

		if X is None:
			X = Variable(torch.zeros(self.num_inputs))

		controller_out = self.controller(X, self.last_read[-1])
		self._read_write(controller_out)

		out = Variable(torch.cat((X, torch.squeeze(self.last_read[-1])), -1))
		out = F.sigmoid(self.fc_out(out))

		return out

	def _read_write(self, controller_out):
		#READ
		mem, w = self.read_head(controller_out, self.memory[-1], self.mem_weights_read[-1])
		self.mem_weights_read.append(w)
		self.last_read.append(mem)

		#WRITE
		mem, w = self.write_head(controller_out, self.memory[-1], self.mem_weights_write[-1])
		self.mem_weights_write.append(w)
		self.memory.append(mem)

	def _initalize_state(self):

		mem_bias = nn.init.uniform_(Variable(torch.Tensor(self.M, self.N)), -1, 1)
		self.memory.append(mem_bias)
		self.mem_weights_read.append(F.softmax(Variable(torch.range(self.M, 1, -1)),dim=-1))
		self.mem_weights_write.append(F.softmax(Variable(torch.range(self.M, 1, -1)), dim=-1))
		self.last_read.append(F.tanh(torch.randn(self.N,)))

	def reset_parameters(self):
		# Initialize the linear layers
		nn.init.xavier_uniform_(self.fc_out.weight, gain=1.4)
		nn.init.normal_(self.fc_out.bias, std=0.01)

	def calculate_num_params(self):
		"""Returns the total number of parameters."""
		num_params = 0
		for p in self.parameters():
			num_params += p.data.view(-1).size(0)
		return num_params
