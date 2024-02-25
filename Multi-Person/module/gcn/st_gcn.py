import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.tgcn import ConvTemporalGraphical,ConvPersonGraphical,ConvGraphical
from .utils.graph import Graph
import numpy as np
from torch.distributions import Gumbel
import random

class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks."""

    def __init__(self, in_channels, hidden_channels, hidden_dim, graph_args,
                 edge_importance_weighting, people_importance_weighting, graph_args_1, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        self.graph_l = Graph(**graph_args_1)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        A_l = torch.tensor(self.graph_l.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.register_buffer('A_l', A_l)

        P_semantic = np.zeros((6, 2))
        P_semantic[[0, 1, 2], [0]] = 1
        P_semantic[[3, 4, 5], [1]] = 1

        P_symmetry = np.zeros((6, 2))
        P_symmetry[[0, 1, 4, 5], [0]] = 1
        P_symmetry[[2, 3], [1]] = 1

        P = np.stack([P_semantic,P_symmetry])
        P = torch.tensor(P, dtype=torch.float32, requires_grad=False)
        # P = torch.unsqueeze(P, 0)
        self.register_buffer('P', P)


        # P = np.zeros((12, 6))
        # for col in range(6):
        #     if col == 0:
        #         P[0:3, col] = 1
        #     elif col == 1:
        #         P[3:6, col] = 1
        #     elif col == 5:
        #         P[9:12, col] = 1
        #     else:
        #         start = col + 2
        #         end = start + 3
        #         P[start:end, col] = 1
        # P = P.reshape(1,12,6)
        # P = torch.tensor(P, dtype=torch.float32, requires_grad=False)
        # self.register_buffer('P', P)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        person_kernel_size = 7
        kernel_size = (temporal_kernel_size, spatial_kernel_size, person_kernel_size)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            # st_gcn(hidden_channels, hidden_channels,hidden_channels, kernel_size, 1, **kwargs),
            # st_gcn(hidden_channels, hidden_channels,hidden_channels, kernel_size, 1, **kwargs),
            # st_gcn(hidden_channels, hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, hidden_channels * 2, kernel_size, 2, **kwargs),
            # st_gcn(hidden_channels * 2, hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            # st_gcn(hidden_channels * 2, hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            # st_gcn(hidden_channels * 4, hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_dim, hidden_dim, kernel_size, 1, **kwargs),

            # st_gcn(in_channels, hidden_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            # st_gcn(hidden_channels, hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            # st_gcn(hidden_channels, hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            # st_gcn(hidden_channels, hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            # st_gcn(hidden_channels, hidden_channels * 2, hidden_channels * 2, kernel_size, 2, **kwargs),
            # st_gcn(hidden_channels * 2, hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            # st_gcn(hidden_channels * 2, hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            # st_gcn(hidden_channels * 2, hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            # st_gcn(hidden_channels * 4, hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
            # st_gcn(hidden_channels * 4, hidden_dim, hidden_dim, kernel_size, 1, **kwargs),
        ))

        self.sgcn_networks = nn.ModuleList((
            sgcn(hidden_dim,kernel_size),
        ))

        self.hgnn_networks = nn.ModuleList((
            hgnn(hidden_dim),
            # hgnn(hidden_dim),
            # hgnn(hidden_dim),
            # hgnn(hidden_dim),
            # hgnn(hidden_dim),
            # hgnn(hidden_dim),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        if people_importance_weighting:
            self.people_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.P.size()))
                for i in self.hgnn_networks
            ])
        else:
            self.people_importance = [1] * len(self.hgnn_networks)

    def forward(self, x, ignore_joint=[], ignore_people=[]):

        # data normalization
        N, C, T, V, M = x.size()

        # Gets the sequence of nodes that are not masked out
        # all_people = set(range(M))
        # remain_people = list(all_people - set(ignore_people))
        # remain_people = sorted(remain_people)
        # x = x[:, :, :, :, remain_people]


        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # Gets the sequence of nodes that are not masked out
        all_joint = set(range(V))
        remain_joint = list(all_joint - set(ignore_joint))
        remain_joint = sorted(remain_joint)
        x = x[:, :, :, remain_joint]


        for gcn, edge_importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * edge_importance, remain_joint)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1)
        # print(x[0,:,:])
        for hgnn, people_importance in zip(self.hgnn_networks, self.people_importance):
            x_1, _ = hgnn(x, self.P, ignore_people)
        # print('hhh',x[0,:,:])
        for sgcn, people_importance in zip(self.sgcn_networks, self.people_importance):
            x_2 = sgcn(x, self.A_l)


        return x_1,x_2



class Model_Linear(nn.Module):
    r"""Spatial temporal graph convolutional networks."""

    def __init__(self, in_channels, hidden_channels, hidden_dim, graph_args,
                 edge_importance_weighting, people_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # P = np.zeros((12, 10))
        # for i in range(1, 11):
        #     P[i - 1:i + 2, i - 1] = 1
        # P = P / P.sum(axis=0)
        P_semantic = np.zeros((6, 2))

        P_semantic[[0, 1, 2], [0]] = 1
        P_semantic[[3, 4, 5], [1]] = 1
        # P_semantic[[6, 7, 8], [2]] = 1
        # P_semantic[[9, 10, 11], [3]] = 1
        P = torch.tensor(P_semantic, dtype=torch.float32, requires_grad=False)
        # P = torch.unsqueeze(P, 0)
        # print(P_semantic)

        P_symmetry = np.zeros((6, 2))

        # P_symmetry[[0, 1, 2, 9, 10, 11], [0]] = 1
        # P_symmetry[[3, 8], [1]] = 1
        # P_symmetry[[4, 7], [2]] = 1
        # P_symmetry[[5, 6], [3]] = 1
        P_symmetry[[0, 1, 4, 5], [0]] = 1
        P_symmetry[[2, 3], [1]] = 1
        # P_symmetry[[4, 7], [2]] = 1
        # P_symmetry[[5, 6], [3]] = 1

        P = np.stack([P_semantic,P_symmetry])
        P = torch.tensor(P, dtype=torch.float32, requires_grad=False)
        P = torch.unsqueeze(P, 0)
        self.register_buffer('P', P)
        
        # P = np.zeros((12, 6))
        # for col in range(6):
        #     if col == 0:
        #         P[0:3, col] = 1
        #     elif col == 1:
        #         P[3:6, col] = 1
        #     elif col == 5:
        #         P[9:12, col] = 1
        #     else:
        #         start = col + 2
        #         end = start + 3
        #         P[start:end, col] = 1
        # P = P.reshape(1,12,6)
        # P = torch.tensor(P, dtype=torch.float32, requires_grad=False)
        # self.register_buffer('P', P)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        person_kernel_size = 7
        kernel_size = (temporal_kernel_size, spatial_kernel_size, person_kernel_size)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            # st_gcn(hidden_channels, hidden_channels,hidden_channels, kernel_size, 1, **kwargs),
            # st_gcn(hidden_channels, hidden_channels,hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels, hidden_channels * 2, hidden_channels * 2, kernel_size, 2, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            # st_gcn(hidden_channels * 2, hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 2, hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            # st_gcn(hidden_channels * 4, hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
            st_gcn(hidden_channels * 4, hidden_dim, hidden_dim, kernel_size, 1, **kwargs),
            # st_gcn(in_channels, hidden_channels, hidden_channels, kernel_size, 1, residual=False, **kwargs0),
            # st_gcn(hidden_channels, hidden_channels,hidden_channels, kernel_size, 1, **kwargs),
            # st_gcn(hidden_channels, hidden_channels,hidden_channels, kernel_size, 1, **kwargs),
            # st_gcn(hidden_channels, hidden_channels, hidden_channels, kernel_size, 1, **kwargs),
            # st_gcn(hidden_channels, hidden_channels * 2, hidden_channels * 2, kernel_size, 2, **kwargs),
            # st_gcn(hidden_channels * 2, hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            # st_gcn(hidden_channels * 2, hidden_channels * 2, hidden_channels * 2, kernel_size, 1, **kwargs),
            # st_gcn(hidden_channels * 2, hidden_channels * 2, hidden_channels * 4, kernel_size, 2, **kwargs),
            # st_gcn(hidden_channels * 4, hidden_channels * 4, hidden_channels * 4, kernel_size, 1, **kwargs),
            # st_gcn(hidden_channels * 4, hidden_dim, hidden_dim, kernel_size, 1, **kwargs),
        ))

        self.hgnn_networks = nn.ModuleList((
            hgnn(hidden_dim),
            hgnn(hidden_dim),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        if people_importance_weighting:
            self.people_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.P.size()))
                for i in self.hgnn_networks
            ])
        else:
            self.people_importance = [1] * len(self.hgnn_networks)

    def forward(self, x, ignore_joint=[], ignore_people=[]):

        # data normalization
        N, C, T, V, M = x.size()

        # Gets the sequence of nodes that are not masked out
        # all_people = set(range(M))
        # remain_people = list(all_people - set(ignore_people))
        # remain_people = sorted(remain_people)
        # x = x[:, :, :, :, remain_people]


        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # Gets the sequence of nodes that are not masked out
        all_joint = set(range(V))
        remain_joint = list(all_joint - set(ignore_joint))
        remain_joint = sorted(remain_joint)
        x = x[:, :, :, remain_joint]


        for gcn, edge_importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * edge_importance, remain_joint)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1)

        for hgnn, people_importance in zip(self.hgnn_networks, self.people_importance):
            x, _ = hgnn(x, self.P, ignore_people)

        x = x.mean(dim=1)

        return x


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert kernel_size[0] % 2 == 1
        padding_t = ((kernel_size[0] - 1) // 2, 0)
        padding_p = ((kernel_size[2] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.hcn = ConvPersonGraphical(out_channels, out_channels)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding_t,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        self.pcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[2], 1),
                (1, 1),
                padding_p,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A, remain_joint):

        A = A[:, remain_joint, :]
        A = A[:, :, remain_joint]
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x)
        x = x + res

        return self.relu(x), A

class sgcn(nn.Module):
    def __init__(self,
                 in_channels,kernel_size):
        super().__init__()

        self.gcn = ConvGraphical(in_channels, in_channels,
                                         kernel_size[1])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        # A = A[:, remain_joint, :]
        # A = A[:, :, remain_joint]
        x, A = self.gcn(x, A)
        return self.relu(x)

        # return self.relu(x)

class hgnn(nn.Module):

    def __init__(self, out_channels):
        super().__init__()

        self.hcn = ConvPersonGraphical(out_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)

        # self.nodevec1 = nn.Parameter(P[0], requires_grad=True)
        # self.nodevec2 = nn.Parameter(P[1].transpose(0,1), requires_grad=True)
        apt_size = 10
        nodevecs = torch.randn(2, 12, apt_size), torch.randn(2, apt_size, 4)
        self.nodevec1, self.nodevec2 = [
            nn.Parameter(n, requires_grad=True) for n in nodevecs
        ]

    def gumbel_softmax(self, x, tau=0.5, hard=False, eps=1e-10):

        gumbels = Gumbel(0, 1).sample(x.shape).cuda()
        y = (gumbels + torch.log(x + eps)) / tau
        y = F.softmax(y, dim=-1)

        if hard:
            _, max_val = torch.max(y, dim=-1, keepdim=True)
            y_hard = y == max_val
            y = (y_hard - y).detach() + y

        return y

    def filter_by_index_v2(self, X1, X2):
        B, L = X1.shape
        T, M, N = X2.shape
        if L != 12:
            indices = X1.long().unsqueeze(-1).repeat(1, 1, N)  # B x L x N
            indices = indices.unsqueeze(1).repeat(1, T, 1, 1)  # B x T x L x N
            X2 = X2.unsqueeze(0).repeat(B, 1, 1, 1)

            output = torch.gather(X2, 2, indices)
        else:
            output = X2.unsqueeze(0).repeat(B, 1, 1, 1)

        return output

    def forward(self, x, P, remain_people):
        import random

        P = self.filter_by_index_v2(remain_people, P)

        length = random.randint(0, 4)
        # 构造概率分布
        dist = [0.1] * 2 + [0.05] * 2 + [0.1] * 2
        indices = random.choices(range(6), weights=dist, k=length)
        # indices = [2,8]
        indices = torch.tensor(indices).cuda()
        # 根据索引置 0
        # print(indices,x.shape,adaptive_adj.shape)

        # adaptive_adj.index_fill_(1, indices, 0)
        # print(type(adaptive_adj),adaptive_adj.dtype,adaptive_adj.shape,x.shape)
        # adaptive_adj = adaptive_adj[:, :, remain_people]

        P[:,:, indices.long(), :] = 0
        x,P = self.hcn(x, P)


        return self.relu(x), P