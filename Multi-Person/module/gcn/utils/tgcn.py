# The based unit of graph convolutional networks.
import torch
import torch.nn as nn

class ConvTemporalGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

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
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A


class ConvGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

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
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()
        self.bn = nn.BatchNorm1d(out_channels*3)
        self.kernel_size = kernel_size
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels,
            out_channels*3,
            kernel_size=(1,),
            padding=0,
            stride=(1,),
            dilation=(1,),
            bias=bias),
        )
        self.eps = nn.Parameter(torch.Tensor([0.]))

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        res = x
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        n, c, m = x.size()
        x = x.contiguous()
        # print('outx',x.shape)
        x = x.view(n, self.kernel_size, c // 3, m)
        # print(x.shape,A.shape)
        x = torch.einsum('nkcm,kme->nce', (x, A))
        x = x.permute(0, 2, 1)
        # x = (1 + self.eps) * res + x
        return x.contiguous(), A







def normalize_l2(X):
    """Row-normalize  matrix"""
    rownorm = X.detach().norm(dim=1, keepdim=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.
    X = X * scale
    return X

class ConvPersonGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

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
                 out_channels,
                 kernel_size=2,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        # self.conv = nn.Conv2d(
        #     in_channels,
        #     out_channels,
        #     kernel_size=(t_kernel_size, 1),
        #     padding=(t_padding, 0),
        #     stride=(t_stride, 1),
        #     dilation=(t_dilation, 1),
        #     bias=bias)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels,
            out_channels*2,
            kernel_size=(1,),
            padding=0,
            stride=(1,),
            dilation=(1,),
            bias=bias),

        )
        self.bn = nn.BatchNorm1d(out_channels*2)
        # conv_layer = nn.Conv1d(in_channels=C, out_channels=3 * C, kernel_size=1)

        self.eps = nn.Parameter(torch.Tensor([0.]))

    def forward(self, x, A):
        # assert A.size(0) == self.kernel_size
        # print(x.shape)
        # res = x
        # _, C, T, V = x.size()
        # x = x.view(N, -1, C, T, V)
        # x = x.permute(0,4,2,3,1)
        # x = x.reshape(N * V, C, T, -1)
        # # print('input',x.shape)
        # x = self.conv(x)
        # # print('outx',x.shape)
        # n, kc, t, v = x.size()
        # x = x.view(n, self.kernel_size, kc//1, t, v)
        # # print(x.shape,A.shape)
        # # x = torch.einsum('nkctv,kvw->nctv', (x, A))
        # x = torch.einsum('nkctv,kvw->nkctw', (x, A))
        # A = A.permute(0,2,1)
        # x = torch.einsum('nkctw,kwv->nkctv', (x, A))
        # x = x.reshape(N, V, C, T, -1)
        # x = x.permute(0,4,2,3,1)
        # x = x.reshape(-1, C, T, V)
        res = x
        x = x.permute(0,2,1)
        # print('input',x.shape)
        x = self.conv(x)
        # print(x.shape)
        x = self.bn(x)
        n, c, m = x.size()
        x = x.contiguous()
        # print('outx',x.shape)
        x = x.view(n, self.kernel_size, c//2, m)
        # print(x.shape,A.shape)
        # x = torch.einsum('nkctv,kvw->nctv', (x, A))
        x = torch.einsum('nkcm,nkme->nkce', (x, A))
        A = A.permute(0,1,3,2)
        # print('2',x.shape,A.shape)
        x = torch.einsum('nkce,nkem->ncm', (x, A))
        x = x.permute(0,2,1)

        x = (1 + self.eps) * res + x
        # print('as',x[0,:,:])
        # x = x + res
        # x = normalize_l2(x)
        
        return x.contiguous(), A