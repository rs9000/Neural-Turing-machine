import torch
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys

import argparse
from ntm import NTM
from logger import Logger

def generate_copy_data(args):

	seq_len = args.sequence_length
	seq_width = args.token_size
	seq = np.random.binomial(1, 0.5, (seq_len, seq_width))
	seq = Variable(torch.from_numpy(seq))

	#Add delimiter token
	inp = Variable(torch.zeros(seq_len + 2, seq_width))
	inp[1:seq_len+1, :seq_width] = seq
	inp[0, 0] = 1.0
	inp[seq_len+1, seq_width-1] = 1.0
	outp = seq.clone()

	return inp.float(), outp.float()


def clip_grads(net):
	parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
	for p in parameters:
		p.grad.data.clamp_(args.min_grad, args.max_grad)


def parse_arguments():

	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--sequence_length', type=int, default=3, help='The length of the sequence to copy', metavar='')
	parser.add_argument('--token_size', type=int, default=10,
						help='The size of the tokens making the sequence', metavar='')
	parser.add_argument('--memory_capacity', type=int, default=64,
						help='Number of records that can be stored in memory', metavar='')
	parser.add_argument('--memory_vector_size', type=int, default=128,
						help='Dimensionality of records stored in memory', metavar='')
	parser.add_argument('--training_samples', type=int, default=999999,
						help='Number of training samples', metavar='')
	parser.add_argument('--controller_output_dim', type=int, default=256,
						help='Dimensionality of the feature vector produced by the controller', metavar='')
	parser.add_argument('--controller_hidden_dim', type=int, default=512,
						help='Dimensionality of the hidden layer of the controller', metavar='')
	parser.add_argument('--learning_rate', type=float, default=1e-4,
						help='Optimizer learning rate', metavar='')
	parser.add_argument('--min_grad', type=float, default=-10.,
						help='Minimum value of gradient clipping', metavar='')
	parser.add_argument('--max_grad', type=float, default=10.,
						help='Maximum value of gradient clipping', metavar='')
	parser.add_argument('--logdir', type=str, default='./logs',
						help='The directory where to store logs', metavar='')

	return parser.parse_args()


if __name__ == "__main__":

	args = parse_arguments()

	logger = Logger(args.logdir)

	model =  NTM(M=args.memory_capacity,
			  N=args.memory_vector_size,
			  num_inputs=args.token_size,
			  num_outputs=args.token_size,
			  controller_out_dim=args.controller_output_dim,
			  controller_hid_dim=args.controller_hidden_dim,
			  learning_rate=args.learning_rate)

	criterion = torch.nn.BCELoss()
	optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)

	print("--------- Number of parameters -----------")
	print(model.calculate_num_params())
	print("--------- Start training -----------")

	losses = []

	#args.training_samples
	for e in range(0, args.training_samples):
		X, Y = generate_copy_data(args)
		inp_seq_len = X.size(0)
		outp_seq_len = Y.size(0)

		optimizer.zero_grad()

		#Input rete: sequenza
		for t in range(0, inp_seq_len):
			model(X[t])

		#Input rete: null
		y_pred = Variable(torch.zeros(Y.size()))
		for i in range(outp_seq_len):
			y_pred[i]= model()

		loss = criterion(y_pred, Y)
		loss.backward()
		clip_grads(model)
		optimizer.step()
		losses += [loss.item()]

		if (e % 500 == 0):
			mean_loss = np.array(losses[-500:]).mean()
			print("Mean Loss: ", mean_loss)

		if(e % 1000 == 0):
			value = loss.item()
			mean_loss = np.array(losses[-1000:]).mean()
			logger.scalar_summary("Mean Loss", mean_loss, e)
			losses = []
