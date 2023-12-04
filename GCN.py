import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from sklearn import preprocessing

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GCNLayer(nn.Module):
    """
    A：　　　　　adjacency matrix
    """

    def __init__(self, input_dim: int, output_dim: int, A: torch.Tensor):
        super(GCNLayer, self).__init__()
        self.A = A + torch.eye(A.shape[0], A.shape[0], requires_grad=False).to(device)
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        # first layer GCN
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 256))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))
        self.mask = torch.ceil(self.A * 0.00001)


    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat

    def forward(self, H, model='normal'):
        H = self.BN(H)
        A = self.A

        D_hat = self.A_to_D_inv(A)
        A_hat = torch.matmul(D_hat, torch.matmul(A,D_hat))
        a = self.GCN_liner_out_1(H)
        output = torch.mm(A_hat, self.GCN_liner_out_1(H))
        output = self.Activition(output)
        return output, A


def dot(H1, H2, epoch):
    (h, w) = H1.shape
    H1_temp = torch.zeros([h, w])
    H2_temp = torch.zeros([h, w])
    cos_map = torch.zeros([h, 1])
    for i in range(h):
        mod = torch.norm(torch.unsqueeze(H1[i], 0), p=2, dim=1) * torch.norm(torch.unsqueeze(H2[i], 0), p=2, dim=1)
        cos = 1 - torch.dot(H1[i], H2[i]) / mod
        cos = math.exp(10 * cos) - 1
        if epoch % 1000 == 0:
            cos_map[i] = cos
        H1_temp[i] = H1[i] * cos
        H2_temp[i] = H2[i] * cos
    minMax = preprocessing.StandardScaler()
    H1_temp = H1_temp.cpu().detach().numpy()
    H2_temp = H2_temp.cpu().detach().numpy()
    H1_temp = minMax.fit_transform(H1_temp)
    H2_temp = minMax.fit_transform(H2_temp)
    H1_temp = torch.from_numpy(H1_temp).to(device)
    H2_temp = torch.from_numpy(H2_temp).to(device)

    return H1_temp, H2_temp, cos_map.to(device)


class CEGCN(nn.Module):
    def __init__(self, height: int, width: int, channel: int, class_count: int, Q1: torch.Tensor, A1: torch.Tensor,
                 Q2: torch.Tensor, A2: torch.Tensor, model='normal'):
        """
        :param height: the height of the image
        :param width:  the width of the image
        :param channel:  the channel of the image
        :param class_count:
        :param Q1: relationship between the image T1 and the superpixel image
        :param A1: adjacency matrix of T1
        :param Q2: relationship between the image T1 and the superpixel image
        :param A2：adjacency matrix of T2
        """
        super(CEGCN, self).__init__()
        # class count
        self.class_count = class_count
        # input data
        self.channel = channel
        self.height = height
        self.width = width
        self.Q1 = Q1
        self.A1 = A1
        self.Q2 = Q2
        self.A2 = A2
        self.model = model
        self.norm_col_Q1 = torch.sum(Q1, 0, keepdim=True)
        self.norm_col_Q2 = torch.sum(Q2, 0, keepdim=True)

        # Superpixel-level Graph Sub-Network
        self.GCN_Branch1 = nn.Sequential()
        self.GCN_Branch1.add_module('GCN_Branch1' + str(0), GCNLayer(self.channel, 512, self.A1))
        self.GCN_Branch1.add_module('GCN_Branch1' + str(1), GCNLayer(512, 1024, self.A1))

        self.GCN_Branch2 = nn.Sequential()
        self.GCN_Branch2.add_module('GCN_Branch2' + str(0), GCNLayer(self.channel, 512, self.A2))
        self.GCN_Branch2.add_module('GCN_Branch2' + str(1), GCNLayer(512, 1024, self.A2))

        self.GCN_Branch_out = nn.Sequential()
        self.GCN_Branch_out.add_module('GCN_Branch_out' + str(0), GCNLayer(2048, 32, self.A1))

        self.Softmax_linear = nn.Sequential(nn.Linear(32, self.class_count))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, epoch: int, curr_seed: int):

        (h1, w1, c1) = x1.shape
        (h2, w2, c2) = x2.shape

        x1_flatten = x1.reshape([h1 * w1, -1])
        superpixels_flatten_1 = torch.mm(self.Q1.t(), x1_flatten)

        x2_flatten = x2.reshape([h2 * w2, -1])
        superpixels_flatten_2 = torch.mm(self.Q2.t(), x2_flatten)

        H1 = superpixels_flatten_1 / self.norm_col_Q1.t()
        H2 = superpixels_flatten_2 / self.norm_col_Q2.t()
        H = []
        if self.model == 'normal':
            for i in range(len(self.GCN_Branch1)):
                H1, _ = self.GCN_Branch1[i](H1)
                H2, _ = self.GCN_Branch2[i](H2)
                if i == 0:
                    H1, H2, cos_map = dot(H1, H2, epoch)
                if i == len(self.GCN_Branch1) - 1:
                    H1, H2, cos_map = dot(H1, H2, epoch)

                    H = torch.cat([H1, H2], dim=1)
                    H, _ = self.GCN_Branch_out[0](H)

        GCN_result = torch.matmul(self.Q1, H)
        Y = self.Softmax_linear(GCN_result)
        Y = F.softmax(Y, -1)
        return Y
