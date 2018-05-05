import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from memory import ReadHead, WriteHead
from controller import Controller


class NTM(nn.Module):
    def __init__(self, M, N, num_inputs, num_outputs, controller_out_dim, controller_hid_dim):
        super(NTM, self).__init__()

        print("----------- Build Neural Turing machine -----------")
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.M = M
        self.N = N

        self.memory = torch.zeros(self.M, self.N)
        self.last_read = torch.zeros(1, self.N)

        self.controller = Controller(self.num_inputs + self.N, controller_out_dim, controller_hid_dim)
        self.read_head = ReadHead(self.M, self.N, controller_out_dim)
        self.write_head = WriteHead(self.M, self.N, controller_out_dim)

        self.fc_out = nn.Linear(self.num_inputs + N, self.num_outputs)
        self.reset_parameters()

    def forward(self, X=None):

        if X is None:
            X = torch.zeros(1, self.num_inputs)

        controller_out = self.controller(X, self.last_read)
        self._read_write(controller_out)

        out = torch.cat((X, self.last_read), -1)
        out = F.sigmoid(self.fc_out(out))

        return out

    def _read_write(self, controller_out):
        # READ
        read, w = self.read_head(controller_out, self.memory)
        self.last_read = read

        # WRITE
        mem, w = self.write_head(controller_out, self.memory)
        self.memory = mem

    def initalize_state(self):
        stdev = 1 / (np.sqrt(self.N + self.M))
        self.memory = nn.init.uniform_((torch.Tensor(self.M, self.N)), -stdev, stdev)
        self.last_read = F.tanh(torch.randn(1, self.N))

        self.read_head.reset_memory()
        self.write_head.reset_memory()

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform_(self.fc_out.weight, gain=1.4)
        nn.init.normal_(self.fc_out.bias, std=0.5)

    def get_memory_info(self):
        return self.memory, self.read_head.get_weights(), self.write_head.get_weights()

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params
