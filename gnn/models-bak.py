#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Pytorch requirements
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
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


class GraphConvolution(nn.Module):
    def __init__(self, nfeat, hidden_dim, name=None, device=None, class_num=None, sparse_inputs=False, act=nn.Tanh, bias=True, dropout=0.0):
        super().__init__()
        self.act = nn.Tanh()
        self.device=device
        self.dropout = dropout
        self.sparse_inputs = sparse_inputs
        self.nfeat = nfeat
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.hidden_dim = 512
        self.class_num = class_num
        self.gcn_weights = nn.Parameter(torch.ones(self.nfeat, self.hidden_dim))
        # if self.bias:
        #     self.gcn_bias = nn.Parameter(torch.zeros(class_num, self.hidden_dim))
        # else:
        #     self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.gcn_weights.size(1))
        self.gcn_weights.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
        #     self.gcn_bias.data.uniform_(-stdv, stdv)

    def forward(self, feat, adj):
        x = feat        #[100, 1024, 101]
        node_size = adj.size()[1]  
        adj = torch.clip(adj, min=0.0)
        I = torch.eye(node_size, device='cuda').unsqueeze(dim=0).to(self.device)
        adj = adj + I      # [1000, m+1, m+1]
        adj = graph_norm_ours(adj, batch=True, self_loop=True, symmetric=True)  #[1000, m+1, m+1]
        x = x.transpose(1, 2)
        pre_sup = torch.matmul(x, self.gcn_weights)  # [m+1, 1000, 1024]
        output = torch.matmul(adj, pre_sup) #[1000, m+1, 1024]

        # if self.bias:
        #     output += self.gcn_bias.unsqueeze(1)
        if self.act is not None:
            return self.act(output[:, 0, :])
        else:
            return output[:, 0, :]


def cal_edge_emb(x, p=2, dim=1):   # v1_graph---taking the similairty by 
    ''' 
    x: (n,K)   [m+1, 1000, 1024]
    return: (n^2, K)
    '''
    x = F.normalize(x, p=p, dim=dim)    #[m+1, 1000, 1024], [100, 1024, 101]
    x_c = x
    x = x.transpose(1, 2)  #[1000, m+1, 1024]  [100, 101, 1024]
    x_r = x  # (K, n, 1) #[1000, m+1, 1024]
    # x_c = torch.transpose(x, 1, 2)  # (K, 1, n) #[1000, 1024, m+1]
    # A = torch.bmm(x_r, x_c).permute(1,2,0)  # (n, n, K) 
    A = torch.bmm(x_r, x_c)     # [1000, m+1, m+1]

    # A = A.view(A.size(0) * A.size(1), A.size(2))  # (n^2, K)
    # print(A.size())
    return A


class GraphNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GraphNN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.fc = nn.Linear(nhid, nclass)

        self.dropout = dropout
        self.cudable = True
        self.decay = 0.3
        self.n_class = nclass
        self.s_centroid = torch.zeros(nclass, nclass)
        self.s_centroid = self.s_centroid.cuda()
        self.t_centroid = torch.zeros(nclass, nclass)
        self.t_centroid = self.t_centroid.cuda()

        self.s1_centroid = torch.zeros(nclass, nhid)
        self.s1_centroid = self.s1_centroid.cuda()
        self.t1_centroid = torch.zeros(nclass, nhid)
        self.t1_centroid = self.t1_centroid.cuda()

        self.MSEloss =  nn.MSELoss()

    def forward(self, x, adj):
        x_gcn = F.relu(self.gc1(x, adj))
        x_gcn = F.dropout(x_gcn, self.dropout, training=self.training)
        gcn_features1 = x_gcn
        x_gcn = self.fc(x_gcn)

        return F.log_softmax(x_gcn, dim=1), gcn_features1


    def loss(self,input,label):

        return -torch.mean(torch.sum(input*label, dim=1))

    def adloss(self, s_feature, t_feature, y_s, y_t):
        n, d = s_feature.shape

        s_labels = y_s

        tlabel = F.softmax(y_t, dim=1)+0 * torch.randn(y_t.size()).cuda()

        # get labels
        t_labels = torch.max(tlabel, 1)[1]


        # image number in each class
        ones_s = torch.ones_like(s_labels, dtype=torch.float)
        ones_t = torch.ones_like(t_labels, dtype=torch.float)
        zeros = torch.zeros(self.n_class)
        if self.cudable:
            zeros = zeros.cuda()
        s_n_classes = zeros.scatter_add(0, s_labels, ones_s)
        t_n_classes = zeros.scatter_add(0, t_labels, ones_t)

        # image number cannot be 0, when calculating centroids
        ones_s = torch.ones_like(s_n_classes)
        ones_t = torch.ones_like(t_n_classes)
        s_n_classes = torch.max(s_n_classes, ones_s)
        t_n_classes = torch.max(t_n_classes, ones_t)

        # calculating centroids, sum and divide
        zeros = torch.zeros(self.n_class, d)
        if self.cudable:
            zeros = zeros.cuda()
        s_sum_feature = zeros.scatter_add(0, torch.transpose(s_labels.repeat(d, 1), 1, 0), s_feature)
        t_sum_feature = zeros.scatter_add(0, torch.transpose(t_labels.repeat(d, 1), 1, 0), t_feature)
        current_s_centroid = torch.div(s_sum_feature, s_n_classes.view(self.n_class, 1))
        current_t_centroid = torch.div(t_sum_feature, t_n_classes.view(self.n_class, 1))

        # Moving Centroid
        decay = self.decay
        s_centroid = (1 - decay) * self.s_centroid + decay * current_s_centroid
        t_centroid = (1 - decay) * self.t_centroid + decay * current_t_centroid
        semantic_loss = self.MSEloss(s_centroid, t_centroid)
        self.s_centroid = s_centroid.detach()
        self.t_centroid = t_centroid.detach()

        return semantic_loss


    def adloss1(self, s_feature, t_feature, y_s, y_t):
        n, d = s_feature.shape

        # get labels
        s_labels = y_s

        tlabel = F.softmax(y_t, dim=1)+0 * torch.randn(y_t.size()).cuda()

        # get labels
        t_labels = torch.max(tlabel, 1)[1]

        # image number in each class
        ones_s = torch.ones_like(s_labels, dtype=torch.float)
        ones_t = torch.ones_like(t_labels, dtype=torch.float)
        zeros = torch.zeros(self.n_class)
        if self.cudable:
            zeros = zeros.cuda()
        s_n_classes = zeros.scatter_add(0, s_labels, ones_s)
        t_n_classes = zeros.scatter_add(0, t_labels, ones_t)

        # image number cannot be 0, when calculating centroids
        ones_s = torch.ones_like(s_n_classes)
        ones_t = torch.ones_like(t_n_classes)
        s_n_classes = torch.max(s_n_classes, ones_s)
        t_n_classes = torch.max(t_n_classes, ones_t)

        # calculating centroids, sum and divide
        zeros = torch.zeros(self.n_class, d)
        if self.cudable:
            zeros = zeros.cuda()
        s_sum_feature = zeros.scatter_add(0, torch.transpose(s_labels.repeat(d, 1), 1, 0), s_feature)
        t_sum_feature = zeros.scatter_add(0, torch.transpose(t_labels.repeat(d, 1), 1, 0), t_feature)
        current_s1_centroid = torch.div(s_sum_feature, s_n_classes.view(self.n_class, 1))
        current_t1_centroid = torch.div(t_sum_feature, t_n_classes.view(self.n_class, 1))

        # Moving Centroid
        decay = self.decay
        s1_centroid = (1 - decay) * self.s1_centroid + decay * current_s1_centroid
        t1_centroid = (1 - decay) * self.t1_centroid + decay * current_t1_centroid
        semantic_loss = self.MSEloss(s1_centroid, t1_centroid)
        self.s1_centroid = s1_centroid.detach()
        self.t1_centroid = t1_centroid.detach()

        return semantic_loss

class MetricNN(nn.Module):
    def __init__(self, args, emb_size):
        super(MetricNN, self).__init__()

        self.gnn_obj = GraphNN(nfeat=512, nhid=512, nclass=10, dropout=0.3)


    def forward(self, z, zi_s, labels_yi):
        # Creating WW matrix
        zero_pad = Variable(torch.zeros(labels_yi[0].size()))
        if self.args.cuda:
            zero_pad = zero_pad.cuda()

        labels_yi = [zero_pad] + labels_yi
        zi_s = [z] + zi_s

        nodes = [torch.cat([zi, label_yi], 1) for zi, label_yi in zip(zi_s, labels_yi)]
        nodes = [node.unsqueeze(1) for node in nodes]
        nodes = torch.cat(nodes, 1)
        adj = cal_edge_emb(nodes)

        logits, gcn_feats = self.gnn_obj(nodes, adj).squeeze(-1)
        outputs = F.sigmoid(logits)

        return outputs, gcn_feats


class SoftmaxModule():
    def __init__(self):
        self.softmax_metric = 'log_softmax'

    def forward(self, outputs):
        if self.softmax_metric == 'log_softmax':
            return F.log_softmax(outputs)
        else:
            raise(NotImplementedError)

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


