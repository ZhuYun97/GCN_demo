import torch as t
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
	
	def __init__(self, in_features, out_features, bias=True):
		super(GraphConvolution, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
		if bias:
			self.bias = nn.Parameter(torch.Tensor(out_features))
		else:
			self.register_parameter('bias', None)

		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

	def forward(self, x, adj):
		support = torch.mm(x, self.weight)
		output = torch.spmm(adj, support) # adj是稀疏矩阵
		if self.bias is not None:
			return output + self.bias
		else:
			return output


class GCN(nn.Module):
	def __init__(self, in_features, nhid, nclass):
		self.in_features = in_features
		self.nhid = nhid
		self.nclass = nclass
		self.gcn1 = GraphConvolution(in_features, nhid)
		self.gcn2 = GraphConvolution(nhid, nclass)

	def forward(self, x, adj):
		h1 = F.relu(self.gcn1(x, adj))
		logits = self.gc2(h1, adj)
		return logits

