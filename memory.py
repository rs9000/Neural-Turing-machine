import torch
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys

class Memory(nn.Module):
	def __init__(self, M, N, controller_out):
		super(Memory, self).__init__()

		self.N = N
		self.M = M
		self.read_lengths = self.N+1+1+3+1
		self.write_lengths = self.N+1+1+3+1+self.N+self.N
		self.controller_out = controller_out

	def size(self):
		return self.N, self.M

	def address(self, k, β, g, s, γ, memory, w_last):

		# Content focus
		wc = self._similarity(k, β, memory)
		# Location focus
		wg = self._interpolate(wc, g, w_last)
		ŵ = self._shift(wg, s)
		w = self._sharpen(ŵ, γ)

		return w

	def _similarity(self, k, β, memory):
		k = k.view(-1, 1)
		#Similarità coseno
		a = torch.matmul(memory,k)
		b = torch.norm(memory) * torch.norm(k) + 1e-16

		w = F.softmax(β * (a / b))
		return w

	def _interpolate(self, wc, g, w_last):
		return g * torch.squeeze(wc) + (1 - g) * w_last

	def _shift(self, wg, s):
		result = Variable(torch.zeros(wg.size()))
		result = _convolve(wg, s)
		return result

	def _sharpen(self, ŵ, γ):
		w = ŵ ** γ
		w = torch.div(w, torch.sum(w, dim=-1).view(-1, 1) + 1e-16)
		return w

class ReadHead(Memory):

	def __init__(self, M, N, controller_out):
		super(ReadHead, self).__init__(M, N, controller_out)

		self.fc_read = nn.Linear(controller_out, self.read_lengths)
		self.reset_parameters()

	def reset_parameters(self):
		# Initialize the linear layers
		nn.init.xavier_uniform(self.fc_read.weight, gain=1.4)
		nn.init.normal(self.fc_read.bias, std=0.01)

	def read(self, memory, w):
		"""Read from memory (according to section 3.1)."""
		return torch.matmul(w.unsqueeze(1), memory).squeeze(1)

	def forward(self,x):

		param = Variable(self.fc_read(x))
		k, β, g, s, γ = torch.split(param,[self.N,1,1,3,1])

		k = F.tanh(k)
		β = F.softplus(β)
		g = F.softplus(g)
		s = F.softmax(s)
		γ = 1+ F.softplus(γ)

		return k, β, g, s, γ


class WriteHead(Memory):

	def __init__(self, M, N, controller_out):
		super(WriteHead, self).__init__(M, N, controller_out)

		self.fc_write = nn.Linear(controller_out, self.write_lengths)
		self.reset_parameters()

	def reset_parameters(self):
		# Initialize the linear layers
		nn.init.xavier_uniform(self.fc_write.weight, gain=1.4)
		nn.init.normal(self.fc_write.bias, std=0.01)

	def write(self, memory, w, e, a):
		"""write to memory (according to section 3.2)."""
		w = w.view(-1,1)
		e = e.view(1,-1)
		a = a.view(1,-1)

		#Moltiplicazione point-wise cazzo!
		erase = torch.mul(w,e)
		add = torch.mul(w,a)
		m_tilde = memory*(1-erase)
		memory_update = m_tilde + add

		return memory_update

	def forward(self,x):

		param = Variable(self.fc_write(x))
		k, β, g, s, γ, a, e = torch.split(param,[self.N,1,1,3,1,self.N,self.N])

		k = F.tanh(k)
		β = F.softplus(β)
		g = F.softplus(g)
		s = F.softmax(s)
		γ = 1+ F.softplus(γ)
		a = F.tanh(a)
		e = F.sigmoid(e)

		return k, β, g, s, γ, a, e


def _convolve(w, s):
	"""Circular convolution implementation."""
	assert s.size(0) == 3
	w = torch.squeeze(w)
	t = torch.cat([w[-1:], w, w[:1]])
	c = F.conv1d(t.view(1, 1, -1), s.view(1, 1,-1)).view(-1)
	return c