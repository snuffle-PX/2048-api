import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from model import nn2048_3

inputdata = Variable(torch.rand(1, 16, 4, 4))

net = nn2048_3()

writer = SummaryWriter(log_dir='./log', comment='nn2048')
with writer:
    writer.add_graph(net, inputdata)
