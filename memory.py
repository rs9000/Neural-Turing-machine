import torch
from torch import nn
import torch.nn.functional as F


class Memory(nn.Module):
    def __init__(self, M, N, controller_out):
        super(Memory, self).__init__()

        self.N = N
        self.M = M
        self.read_lengths = self.N + 1 + 1 + 3 + 1
        self.write_lengths = self.N + 1 + 1 + 3 + 1 + self.N + self.N
        self.w_last = []
        self.reset_memory()

    def get_weights(self):
        return self.w_last

    def reset_memory(self):
        self.w_last = []
        self.w_last.append(torch.zeros([1, self.M], dtype=torch.float32))

    def address(self, k, β, g, s, γ, memory, w_last):
        # Content focus
        wc = self._similarity(k, β, memory)
        # Location focus
        wg = self._interpolate(wc, g, w_last)
        ŵ = self._shift(wg, s)
        w = self._sharpen(ŵ, γ)

        return w

    def _similarity(self, k, β, memory):
        # Similarità coseno
        w = F.cosine_similarity(memory, k, -1, 1e-16)
        w = F.softmax(β * w, dim=-1)
        return w

    def _interpolate(self, wc, g, w_last):
        return g * wc + (1 - g) * w_last

    def _shift(self, wg, s):
        result = torch.zeros(wg.size())
        result = _convolve(wg, s)
        return result

    def _sharpen(self, ŵ, γ):
        w = ŵ ** γ
        w = torch.div(w, torch.sum(w, dim=-1) + 1e-16)
        return w


class ReadHead(Memory):

    def __init__(self, M, N, controller_out):
        super(ReadHead, self).__init__(M, N, controller_out)

        print("--- Initialize Memory: ReadHead")
        self.fc_read = nn.Linear(controller_out, self.read_lengths)
        self.reset_parameters();

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform_(self.fc_read.weight, gain=1.4)
        nn.init.normal_(self.fc_read.bias, std=0.01)

    def read(self, memory, w):
        """Read from memory (according to section 3.1)."""
        return torch.matmul(w, memory)

    def forward(self, x, memory):
        param = self.fc_read(x)
        k, β, g, s, γ = torch.split(param, [self.N, 1, 1, 3, 1], dim=1)

        k = F.tanh(k)
        β = F.softplus(β)
        g = F.sigmoid(g)
        s = F.softmax(s, dim=1)
        γ = 1 + F.softplus(γ)

        w = self.address(k, β, g, s, γ, memory, self.w_last[-1])
        self.w_last.append(w)
        mem = self.read(memory, w)
        return mem, w


class WriteHead(Memory):

    def __init__(self, M, N, controller_out):
        super(WriteHead, self).__init__(M, N, controller_out)

        print("--- Initialize Memory: WriteHead")
        self.fc_write = nn.Linear(controller_out, self.write_lengths)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform_(self.fc_write.weight, gain=1.4)
        nn.init.normal_(self.fc_write.bias, std=0.01)

    def write(self, memory, w, e, a):
        """write to memory (according to section 3.2)."""
        w = torch.squeeze(w)
        e = torch.squeeze(e)
        a = torch.squeeze(a)

        erase = torch.ger(w, e)
        add = torch.ger(w, a)

        m_tilde = memory * (1 - erase)
        memory_update = m_tilde + add

        return memory_update

    def forward(self, x, memory):
        param = self.fc_write(x)

        k, β, g, s, γ, a, e = torch.split(param, [self.N, 1, 1, 3, 1, self.N, self.N], dim=1)

        k = F.tanh(k)
        β = F.softplus(β)
        g = F.sigmoid(g)
        s = F.softmax(s, dim=-1)
        γ = 1 + F.softplus(γ)
        a = F.tanh(a)
        e = F.sigmoid(e)

        w = self.address(k, β, g, s, γ, memory, self.w_last[-1])
        self.w_last.append(w)
        mem = self.write(memory, w, e, a)
        return mem, w


def _convolve(w, s):
    """Circular convolution implementation."""
    b, d = s.shape
    assert b == 1, 'does _convolve work for b != 1?'
    assert d == 3
    w = torch.squeeze(w)
    t = torch.cat([w[-1:], w, w[:1]])
    c = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(b, -1)
    return c
