import torch
import numpy as np
import argparse
from ntm import NTM
from time import time
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class BinaySeqDataset(Dataset):

    def __init__(self, args):
        self.seq_len = args.sequence_length
        self.seq_width = args.token_size
        self.dataset_dim = args.training_samples

    def _generate_seq(self):
        seq = np.random.binomial(1, 0.5, (self.seq_len, self.seq_width))
        seq = torch.from_numpy(seq)
        # Add start and end token
        inp = torch.zeros(self.seq_len + 2, self.seq_width)
        inp[1:self.seq_len + 1, :self.seq_width] = seq.clone()
        inp[0, 0] = 1.0
        inp[self.seq_len + 1, self.seq_width - 1] = 1.0
        outp = seq.data.clone()

        return inp.float(), outp.float()

    def __len__(self):
        return self.dataset_dim

    def __getitem__(self, idx):
        inp, out = self._generate_seq()
        return inp, out


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
    parser.add_argument('--logdir', type=str, default='./logs2',
                        help='The directory where to store logs', metavar='')
    parser.add_argument('--loadmodel', type=str, default='checkpoint/checkpoint.model',
                        help='The pre-trained model checkpoint', metavar='')
    parser.add_argument('--savemodel', type=str, default='checkpoint.model',
                        help='Name/Path of model checkpoint', metavar='')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    writer = SummaryWriter()
    dataset = BinaySeqDataset(args)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, num_workers=4)

    model = NTM(M=args.memory_capacity,
                N=args.memory_vector_size,
                num_inputs=args.token_size,
                num_outputs=args.token_size,
                controller_out_dim=args.controller_output_dim,
                controller_hid_dim=args.controller_hidden_dim,
                )

    print(model)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)

    print("--------- Number of parameters -----------")
    print(model.calculate_num_params())
    print("--------- Start training -----------")

    losses = []

    if args.loadmodel != '':
        model.load_state_dict(torch.load(args.loadmodel))

    for e, (X, Y) in enumerate(dataloader):
        tmp = time()
        model.initalize_state()
        optimizer.zero_grad()

        inp_seq_len = args.sequence_length + 2
        out_seq_len = args.sequence_length

        X.requires_grad = True

        # Input rete: sequenza
        for t in range(0, inp_seq_len):
            model(X[:, t])

        # Input rete: null
        y_pred = torch.zeros(Y.size())
        for i in range(0, out_seq_len):
            y_pred[:, i] = model()

        loss = criterion(y_pred, Y)
        loss.backward()
        clip_grads(model)
        optimizer.step()
        losses += [loss.item()]

        if e % 50 == 0:
            mean_loss = np.array(losses[-50:]).mean()
            print("Loss: ", loss.item())
            writer.add_scalar('Mean loss', loss.item(), e)
            if e % 1000 == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), e)
                mem_pic, read_pic, write_pic = model.get_memory_info()
                pic1 = vutils.make_grid(y_pred, normalize=True, scale_each=True)
                pic2 = vutils.make_grid(Y, normalize=True, scale_each=True)
                pic3 = vutils.make_grid(mem_pic, normalize=True, scale_each=True)
                pic4 = vutils.make_grid(read_pic, normalize=True, scale_each=True)
                pic5 = vutils.make_grid(write_pic, normalize=True, scale_each=True)
                writer.add_image('NTM output', pic1, e)
                writer.add_image('True output', pic2, e)
                writer.add_image('Memory', pic3, e)
                writer.add_image('Read weights', pic4, e)
                writer.add_image('Write weights', pic5, e)
                torch.save(model.state_dict(), args.savemodel)
            losses = []