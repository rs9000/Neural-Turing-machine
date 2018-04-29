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


def generate_copy_data(args):
	data = []
	for _ in range(0, args.sequence_length):
		data.append(np.round(np.random.rand(args.token_size).astype('f')))

	start_token = np.zeros(shape=(args.token_size,), dtype=np.float32)
	start_token[0] = 1
	start_token = torch.from_numpy(start_token)

	end_token = np.zeros(shape=(args.token_size,), dtype=np.float32)
	end_token[1] = 1
	end_token = torch.from_numpy(end_token)

	X = np.stack(data, axis=0)
	Y = X.copy()

	X_list = []
	Y_list = []

	for t in range(0, args.sequence_length):
		X_list.append(torch.from_numpy(X[t]))

	for t in range(0, args.sequence_length):
		Y_list.append(torch.from_numpy(Y[t]))

	zeros = torch.from_numpy(np.zeros(shape=(args.token_size,), dtype=np.float32))
	return start_token, X_list, Y_list, end_token, zeros


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
	parser.add_argument('--logdir', type=str, default='logs',
						help='The directory where to store logs', metavar='')

	return parser.parse_args()


if __name__ == "__main__":

	args = parse_arguments()

	model =  NTM(M=args.memory_capacity,
			  N=args.memory_vector_size,
			  num_inputs=args.token_size,
			  sequence_length=args.sequence_length,
			  controller_out_dim=args.controller_output_dim,
			  controller_hid_dim=args.controller_hidden_dim,
			  learning_rate=args.learning_rate)

	criterion = torch.nn.BCELoss(size_average=True)
	optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)

	print("--------- Number of parameters -----------")
	print(model.calculate_num_params())

	for e in range(0, args.training_samples):
		s_token, X, Y, e_token, zeros = generate_copy_data(args)
		y_pred = model(s_token,X,e_token,zeros)

		loss = 0
		optimizer.zero_grad()

		for t in range(0, args.sequence_length):
			loss += criterion(y_pred[t],Y[t])

		if (e % 500 == 0):
			print("Loss: ", loss.item())

		loss.backward(retain_graph=True)
		optimizer.step()