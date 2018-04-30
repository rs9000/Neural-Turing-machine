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
	def __init__(self, M, N, num_inputs, sequence_length, controller_out_dim, controller_hid_dim, learning_rate):
		super(NTM, self).__init__()

		print("Build Neural Turing machine\n")
		self.num_inputs = num_inputs
		self.num_outputs = num_inputs-1
		self.M = M
		self.N = N
		self.sequence_length = sequence_length
		self.learning_rate = learning_rate
		self.controller_out_dim = controller_out_dim

		self.controller = Controller(num_inputs+N, self.controller_out_dim, controller_hid_dim)
		self.read_head = ReadHead(self.M, self.N, self.controller_out_dim)
		self.write_head = WriteHead(self.M, self.N, self.controller_out_dim)

		self.outputs = []
		self.memory = []
		self.last_read = []
		self.mem_weights_read = []
		self.mem_weights_write = []

		self.X = []
		self.Y = []

		self.fc_out = nn.Linear(num_inputs+N, self.num_outputs)
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
		# Step: Genera parametri, genera pesi, reading
		k, β, g, s, γ = self.read_head(controller_out)
		self.mem_weights_read.append(self.read_head.address(k, β, g, s, γ, self.memory[-1], self.mem_weights_read[-1]))
		self.last_read.append(self.read_head.read(self.memory[-1], self.mem_weights_read[-1]))

		#WRITE
		# Step: Genera parametri, genera pesi, writing
		k, β, g, s, γ, a, e = self.write_head(controller_out)
		self.mem_weights_write.append(self.read_head.address(k, β, g, s, γ, self.memory[-1], self.mem_weights_write[-1]))
		memory_update = self.write_head.write(self.memory[-1], self.mem_weights_write[-1], e , a)
		self.memory.append(memory_update)

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
