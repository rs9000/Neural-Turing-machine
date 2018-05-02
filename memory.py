import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import numpy as np

class NTMMemory(nn.Module):

	def __init__(self, N, M):
		#Inizializzazione memoria dim NxM
		super(NTMMemory, self).__init__()

		self.N = N
		self.M = M

		# The memory bias allows the heads to learn how to initially address
		# memory locations by content

		self.mem_bias = Variable(torch.Tensor(N, M))
		nn.init.uniform(self.mem_bias, -1, 1)

	def reset(self, batch_size):
		"""Initialize memory from bias, for start-of-sequence."""
		self.batch_size = batch_size
		self.memory = self.mem_bias.clone().repeat(batch_size, 1, 1)

	def size(self):
		return self.N, self.M

	def read(self, w):
		#Read (paper 3.1)
		#Moltiplico i pesi per la memoria
		return torch.matmul(w.unsqueeze(1), self.memory).squeeze(1)

	def write(self, w, e, a):
		#Write (paper 3.2)
		self.prev_mem = self.memory
		self.memory = Variable(torch.Tensor(self.batch_size, self.N, self.M))
		erase_step = self.prev_mem * (1 - (torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))))
		add_step = erase_step + torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
		self.memory = add_step

	def address(self, k, β, g, s, γ, w_prev):
		#Addressing (paper 3.3)

		# Content focus
		wc = self._similarity(k, β)
		# Location focus
		wg = self._interpolate(w_prev, wc, g)
		ŵ = self._shift(wg, s)
		w = self._sharpen(ŵ, γ)

		return w

	def _similarity(self, k, β):
		k = k.view(self.batch_size, 1, -1)

		sim = F.cosine_similarity(self.memory,k,-1,1e-16)
		w = F.softmax(β * sim, dim=1)
		print(w)
		return w

	def _interpolate(self, w_prev, wc, g):
		return g * wc + (1 - g) * w_prev

	def _shift(self, wg, s):
		result = Variable(torch.zeros(wg.size()))
		for b in range(self.batch_size):
			result[b] = _convolve(wg[b], s[b])
		return result

	def _sharpen(self, ŵ, γ):
		w = ŵ ** γ
		w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
		return w


def _convolve(w, s):
		assert s.size(0) == 3
		t = torch.cat([w[-1:], w, w[:1]])
		c = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(-1)
		return c