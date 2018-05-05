import torch
import numpy as np
import argparse
from ntm import NTM
from time import time
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

def generate_copy_data(args):
    seq_len = args.sequence_length
    seq_width = args.token_size
    seq = np.random.binomial(1, 0.5, (seq_len, seq_width))
    seq = torch.from_numpy(seq)
    seq.requires_grad = True

    # Add delimiter token
    inp = torch.zeros(seq_len + 2, seq_width)
    inp[1:seq_len + 1, :seq_width] = seq.clone()
    inp[0, 0] = 1.0
    inp[seq_len + 1, seq_width - 1] = 1.0
    outp = seq.data.clone()

    # Add batch singleton dimension
    inp = torch.unsqueeze(inp, dim=0)
    outp = torch.unsqueeze(outp, dim=0)

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
    parser.add_argument('--logdir', type=str, default='./logs2',
                        help='The directory where to store logs', metavar='')
    parser.add_argument('--loadmodel', type=str, default='',
                        help='The pre-trained model checkpoint', metavar='')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    writer = SummaryWriter()

    model = NTM(M=args.memory_capacity,
                N=args.memory_vector_size,
                num_inputs=args.token_size,
                num_outputs=args.token_size,
                controller_out_dim=args.controller_output_dim,
                controller_hid_dim=args.controller_hidden_dim,
                )

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)

    print("--------- Number of parameters -----------")
    print(model.calculate_num_params())
    print("--------- Start training -----------")

    losses = []

    if args.loadmodel != '':
        model.load_state_dict(torch.load(args.loadmodel))

    # args.training_samples
    for e in range(0, args.training_samples):
        tmp = time()
        model.initalize_state()
        optimizer.zero_grad()

        inp_seq_len = args.sequence_length + 2
        out_seq_len = args.sequence_length

        X, Y = generate_copy_data(args)

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

        if (e % 50 == 0):
            mean_loss = np.array(losses[-50:]).mean()
            print("Loss: ", loss.item())
            writer.add_scalar('Mean loss', loss.item(), e)
            for name, param in model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), e)
            if (e % 1000 == 0):
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
                torch.save(model.state_dict(), "checkpoint.model")
            losses = []