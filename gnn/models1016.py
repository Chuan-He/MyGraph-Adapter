import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_l = torch.cuda.LongTensor


def graph_norm_ours(A, batch=False, self_loop=True, symmetric=True):
	# A = A + I    A: (bs, num_nodes, num_nodes
    # Degree
    d = A.sum(-1) # (bs, num_nodes) #[1000, m+1]
    if symmetric:
		# D = D^-1/2
        d = torch.pow(d, -0.5)
        if batch:
            D = A.detach().clone()
            for i in range(A.size(0)):
                D[i] = torch.diag(d[i])
            norm_A = D.bmm(A).bmm(D)
        else:
            D = torch.diag(d)
            norm_A = D.mm(A).mm(D)
    else:
		# D=D^-1
        d = torch.pow(d,-1)
        if batch:
            D = A.detach().clone()
            for i in range(A.size(0)):
                D[i] = torch.diag(d[i])
            norm_A = D.bmm(A)
        else:
            D =torch.diag(d)
            norm_A = D.mm(A)

    return norm_A

class Wcompute(nn.Module):
    def __init__(self, input_features, nf, operator='laplace', activation='softmax', drop=False):
        super(Wcompute, self).__init__()
        self.num_features = nf
        self.operator = operator
        self.conv2d_1 = nn.Conv2d(input_features, nf, 1, stride=1)
        self.conv2d_last = nn.Conv2d(nf, 1, 1, stride=1)
        self.bn_1 = nn.BatchNorm2d(nf)
        self.drop = drop
        if self.drop:
            self.dropout = nn.Dropout(0.3)
        self.activation = activation

    def forward(self, x, W_id):
        W1 = x.unsqueeze(2)
        W2 = torch.transpose(W1, 1, 2) #size: bs x N x N x num_features
        W_new = torch.abs(W1 - W2) #size: bs x N x N x num_features
        W_new = torch.transpose(W_new, 1, 3) #size: bs x num_features x N x N

        W_new = self.conv2d_1(W_new)
        W_new = self.bn_1(W_new)
        W_new = F.leaky_relu(W_new)
        if self.drop:
            W_new = self.dropout(W_new)

        W_new = self.conv2d_last(W_new)
        W_new = torch.transpose(W_new, 1, 3) #size: bs x N x N x 1

        if self.activation == 'softmax':
            W_new = W_new - W_id.expand_as(W_new) * 1e8
            W_new = torch.transpose(W_new, 2, 3)
            # Applying Softmax
            W_new = W_new.contiguous()
            W_new_size = W_new.size()
            W_new = W_new.view(-1, W_new.size(3))
            W_new = F.softmax(W_new)
            W_new = W_new.view(W_new_size)
            # Softmax applied
            W_new = torch.transpose(W_new, 2, 3)

        elif self.activation == 'sigmoid':
            W_new = F.sigmoid(W_new)
            W_new *= (1 - W_id)
        elif self.activation == 'none':
            W_new *= (1 - W_id)
        else:
            raise (NotImplementedError)

        if self.operator == 'laplace':
            W_new = W_id - W_new
        elif self.operator == 'J2':
            W_new = torch.cat([W_id, W_new], 3)
        else:
            raise(NotImplementedError)

        return W_new.squeeze(-1)
    
class GraphConvolution(nn.Module):
    def __init__(self, nfeat, output_dim, bias=True, dropout=0.0):
        super().__init__()
        self.act = nn.Tanh()
        self.dropout = dropout
        self.nfeat = nfeat
        self.output_dim = output_dim
        self.bias = bias
        #self.class_num = class_num
        self.gcn_weights = nn.Parameter(torch.ones(self.nfeat, self.output_dim, device='cuda'))
        if self.bias:
            self.gcn_bias = nn.Parameter(torch.zeros(self.output_dim, device='cuda'))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.gcn_weights.size(1))
        self.gcn_weights.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.gcn_bias.data.uniform_(-stdv, stdv)

    def forward(self, feat, adj):
        x = feat        #[100, 1024, 101]
        node_size = adj.size()[1]  
        adj = torch.clip(adj, min=0.0)
        I = torch.eye(node_size, device='cuda').unsqueeze(dim=0).to('cuda')
        adj = adj + I      # [1000, m+1, m+1]
        #adj = graph_norm_ours(adj, batch=True, self_loop=True, symmetric=True)  #[1000, m+1, m+1]
        #x = x.transpose(1, 2)
        pre_sup = torch.matmul(x, self.gcn_weights)  # [m+1, 1000, 1024]
        output = torch.matmul(adj, pre_sup) #[1000, m+1, 1024]

        if self.bias:
            output += self.gcn_bias.unsqueeze(0)
        if self.act is not None:
            return self.act(output)
        else:
            return output
        
    # def forward_text(self, input, adj):
    #     support = torch.mm(input, self.gcn_weights)
    #     output = torch.spmm(adj, support)
    #     if self.bias:
    #         output += self.gcn_bias.unsqueeze(0)
    #     if self.act is not None:
    #         return self.act(output)
    #     else:
    #         return output


class GraphNN(nn.Module):
    def __init__(self, nfeat):
        super(GraphNN, self).__init__()
        self.MetricNN = Wcompute(nfeat, nfeat, operator='laplace', activation='softmax')
        self.GCov = GraphConvolution(nfeat=nfeat, output_dim=nfeat)
        # self.gc2 = GraphConvolution(nfeat=nfeat, output_dim=nhid)
        #self.fc = nn.Linear(nhid, nclass, device='cuda')

    def forward(self, x):
        W_init = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3)).cuda()

        W = self.MetricNN(x, W_init)

        x_new = F.leaky_relu(self.GCov(x, W))

        return x_new


if __name__ == '__main__':
    # test modules
    bs =  4
    nf = 10
    num_layers = 5
    N = 8
    x = torch.ones((bs, N, nf))
    W1 = torch.eye(N).unsqueeze(0).unsqueeze(-1).expand(bs, N, N, 1)
    W2 = torch.ones(N).unsqueeze(0).unsqueeze(-1).expand(bs, N, N, 1)
    J = 2
    W = torch.cat((W1, W2), 3)
    input = [Variable(W), Variable(x)]
    ######################### test gmul ##############################
    # feature_maps = [num_features, num_features, num_features]
    # out = gmul(input)
    # print(out[0, :, num_features:])
    ######################### test gconv ##############################
    # feature_maps = [num_features, num_features, num_features]
    # gconv = Gconv(feature_maps, J)
    # _, out = gconv(input)
    # print(out.size())
    ######################### test gnn ##############################
    # x = torch.ones((bs, N, 1))
    # input = [Variable(W), Variable(x)]
    # gnn = GNN(num_features, num_layers, J)
    # out = gnn(input)
    # print(out.size())



