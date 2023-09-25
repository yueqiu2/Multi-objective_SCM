import torch
import numpy as np

class Net(torch.nn.Module):

    def __init__(self, num_nodes, length, trans_type):
        super(Net, self).__init__()
        self.dtype    = torch.float

        # Get info of machine
        self.use_cuda = torch.cuda.is_available()
        self.device   = torch.device("cpu")

        self.num_nodes = num_nodes
        self.num_trans_type = trans_type

        # Topology
        self.input_size = num_nodes * length
        self.output_size_gen = num_nodes * 4
        self.output_size_reorder = num_nodes
        self.output_size_trans = num_nodes * trans_type

        # Layers - 1 hidden 1*input
        self.hidden1 = torch.nn.Linear(self.input_size, self.input_size)
        self.output_reorder = torch.nn.Linear(self.input_size, self.output_size_reorder)
        self.output_trans = torch.nn.Linear(self.input_size, self.output_size_trans)

        # # Layers - 1 hidden 2*input
        # self.hidden1 = torch.nn.Linear(self.input_size, self.input_size * 2)
        # self.output_reorder = torch.nn.Linear(self.input_size * 2, self.output_size_reorder)
        # self.output_trans = torch.nn.Linear(self.input_size * 2, self.output_size_trans)

        # # Layers -  2 hidden
        # self.hidden1 = torch.nn.Linear(self.input_size, self.input_size)
        # self.hidden2 = torch.nn.Linear(self.input_size, self.output_size_gen)
        # self.output_reorder = torch.nn.Linear(self.output_size_gen, self.output_size_reorder)
        # self.output_trans = torch.nn.Linear(self.output_size_gen, self.output_size_trans)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(1, -1).float()
        y = torch.nn.functional.leaky_relu(self.hidden1(x), 0.1)
        # y = torch.nn.functional.leaky_relu(self.hidden2(y), 0.1)

        reorder = torch.nn.functional.relu6(self.output_reorder(y)) * 50 / 6
        reorder = torch.round(reorder)

        # Get transportation types
        trans_logits = self.output_trans(y).view(self.num_nodes, self.num_trans_type)
        trans_prob = self.softmax(trans_logits)
        trans_choice = torch.argmax(trans_prob, dim=1)  # Get the index of the max probability

        return (reorder, trans_choice)