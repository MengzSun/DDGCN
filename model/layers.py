from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import copy
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch_sparse import spmm
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean
# from utils import *

#raw GCN
class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv,stdv)

    def forward(self, feature, adj):
        #        support = torch.spmm(feature, self.weight) # sparse
        #        output = torch.spmm(adj, support)
        support = torch.mm(feature, self.weight)  # sparse
        # print('support:',support.size())
        output = spmm(adj._indices(), adj._values(), adj.size(0),adj.size(1), support)
        # TypeError: spmm() missing 1 required positional argument: 'matrix'(少一个参数)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphConvolution_nBatch(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution_nBatch, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv,stdv)

    def forward(self, x, edge_index,edge_weight):
        #        support = torch.spmm(feature, self.weight) # sparse
        #        output = torch.spmm(adj, support)
        temp_shape = torch.zeros(x.size(0),x.size(0))
        shape = temp_shape.size()
        # print('layers edge index:',edge_index)
        # print('layers edge weight:', edge_weight)
        # print('shape:',shape)
        adj_sparse = torch.sparse.FloatTensor(edge_index, edge_weight, shape)
        # print('adj sparse size:',adj_sparse.size())
        support = torch.mm(x, self.weight)  # sparse
        # print('support:',support.size())
        output = spmm(adj_sparse._indices(), adj_sparse._values(), adj_sparse.size(0),adj_sparse.size(1), support)
        # TypeError: spmm() missing 1 required positional argument: 'matrix'(少一个参数)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

#geometric propagation GCN
class PropagationGraphConvLayer(Module):
    def __init__(self, in_features,hidden_features, out_features,dropout,device, bias=True):
        super(PropagationGraphConvLayer, self).__init__()
        self.conv1 = GraphConvolution(in_features,hidden_features)
        self.conv2 = GraphConvolution(hidden_features+in_features, out_features)
        self.linear = nn.Linear(hidden_features+out_features,in_features)
        self.dropout = dropout
        self.device = device
    def forward(self, features, adjs, root_idx):
        # print('feature',features.size())
        features_1 = copy.copy(features.float())
        # print('features:',features.size())
        # print('adjs:',adjs.size())
        features = self.conv1(features, adjs)
        features_2 = copy.copy(features)
        root_extend = torch.zeros(features_1.size(0), features_1.size(1)).to(self.device)
        # print('root_extend:',root_extend.size())
        for index in range(features_1.size(0)):
            root_extend[index] = features_1[root_idx]
        features = torch.cat([features,root_extend],1)

        features = F.relu(features)
        features = F.dropout(features,self.dropout,training=self.training)
        features = self.conv2(features, adjs)
        features = F.relu(features)
        root_extend = torch.zeros(features_2.size(0),features_2.size(1)).to(self.device)
        for index in range(features_2.size(0)):
            root_extend[index] = features_2[root_idx]
        features = torch.cat([features,root_extend],1)
        features = F.leaky_relu(self.linear(features))
        return features

class PropagationGraphConvLayer_nBatch(Module):
    def __init__(self, in_features,hidden_features, out_features,dropout,device, bias=True):
        super(PropagationGraphConvLayer_nBatch, self).__init__()
        # self.conv1 = GCNConv(in_features,hidden_features,add_self_loops=False,normalize=False)
        # self.conv2 = GCNConv(hidden_features+in_features, out_features, add_self_loops=False,normalize=False)
        self.conv1 = GraphConvolution_nBatch(in_features, hidden_features)
        self.conv2 = GraphConvolution_nBatch(hidden_features + in_features, out_features)
        self.linear = nn.Linear(hidden_features+out_features,in_features)
        self.dropout = dropout
        self.device = device
    def forward(self, features, adjs, values, root_idx, propagation_node_num, propagation_edge_num):
        # print('feature',features.size())
        # print('----layer features----:', features.size())
        # print('----layer adjs----:', adjs.size())
        # print('----layer values----:', values.size())
        # print('----layer root_idx---:',root_idx)
        # print('----layer propagation node num---:',propagation_node_num)
        # print('----layer propagation edge num---:', propagation_edge_num)
        print('----layer adjs----:', adjs)
        node_num_count = 0
        edge_num_count = 0
        for i in range(len(propagation_node_num)):
            if i == 0:
                indices_slect = torch.LongTensor(range(propagation_edge_num[i].item())).to(self.device)
                features_short = features[0:propagation_node_num[i].item()]
                features_1 = copy.copy(features_short.float())
                # print('indices:',indices_slect)
                # print('adjs:',adjs)
                adjs_short = torch.index_select(adjs,1,indices_slect).to(self.device)
                values_short = torch.index_select(values,0,indices_slect).to(self.device)
                # adjs_short = adjs[:][0:propagation_edge_num[i].item()]
                # values_short = values[0:propagation_edge_num[i].item()]
                print('----layer features short  0----:',features_short.size())
                print('----layer adjs short  0----:',adjs_short.size())
                print('----layer adjs short  0----:', adjs_short)
                print('----layer values short  0----:',values_short.size())
                print('----layer edge num  0----:',propagation_edge_num[i].item())
                print('----layer node num  0----:', propagation_node_num[i].item())
                features_short = self.conv1(x=features_short,edge_index=adjs_short,edge_weight=values_short)
                features_2 = copy.copy(features_short)
                root_extend = torch.zeros(features_1.size(0), features_1.size(1)).to(self.device)
                # print('root_extend:',root_extend.size())
                # print('len features 1:',list(range(features_1.size(0))))
                # indices_slect_1 = torch.LongTensor(range(root_extend.size(0)))
                for index in list(range(root_extend.size(0))):
                    # print('index:',index)
                    # print('root_idx:',root_idx[i])
                    root_extend[index] = features_1[root_idx[i]]
                features_short = torch.cat([features_short,root_extend],1)

                features_short = F.relu(features_short)
                features_short = F.dropout(features_short,self.dropout,training=self.training)
                features_short = self.conv2(x=features_short, edge_index=adjs_short,edge_weight=values_short)
                features_short = F.relu(features_short)
                root_extend = torch.zeros(features_2.size(0),features_2.size(1)).to(self.device)
                for index in range(features_2.size(0)):
                    root_extend[index] = features_2[root_idx[i]]
                features_short = torch.cat([features_short,root_extend],1)
                features_short = F.leaky_relu(self.linear(features_short))
                features_all = features_short

                node_num_count += propagation_node_num[i].item()
                edge_num_count += propagation_edge_num[i].item()
            else:
                indices_slect = torch.LongTensor(range(edge_num_count,edge_num_count+propagation_edge_num[i].item())).to(self.device)
                features_short = features[node_num_count:node_num_count+propagation_node_num[i].item()]
                features_1 = copy.copy(features_short.float())
                # print('features_1:',features_1.size())
                # print('adjs:',adjs.size())
                adjs_short = torch.index_select(adjs, 1, indices_slect).to(self.device)
                values_short = torch.index_select(values, 0, indices_slect).to(self.device)
                # adjs_short = adjs[:][edge_num_count:edge_num_count+propagation_edge_num[i].item()]
                # values_short = values[edge_num_count:edge_num_count+propagation_edge_num[i].item()]
                print('----layer features short ----:', features_short.size())
                print('----layer adjs short  ----:', adjs_short.size())
                print('----layer adjs short  ----:', adjs_short)
                print('----layer values short  ----:', values_short.size())
                print('----layer edge num  ----:', propagation_edge_num[i].item())
                print('----layer node num  ----:', propagation_node_num[i].item())
                print('----layer node num  count----:', node_num_count)
                for j in range(adjs_short.size(1)):
                    adjs_short[0][j] = adjs_short[0][j]-edge_num_count
                    adjs_short[1][j] = adjs_short[1][j] - edge_num_count
                # print('----layer features short----:',features_short.size())
                # print('----layer adjs short----:',adjs_short.size())
                # print('----layer adjs short----:', adjs_short)
                # print('----layer values short----:',values_short.size())
                # print('----layer edge num----:',propagation_edge_num[i].item())
                # print('----layer node num----:', propagation_node_num[i].item())
                features_short = self.conv1(x=features_short,edge_index=adjs_short,edge_weight=values_short)
                features_2 = copy.copy(features_short)
                root_extend = torch.zeros(features_1.size(0), features_1.size(1)).to(self.device)
                # print('root_extend:',root_extend.size())
                for index in range(features_1.size(0)):
                    root_idx_new = root_idx[i]-node_num_count
                    # print('index:',index)
                    # print('root_idx:',root_idx_new)
                    # return 0
                    root_extend[index] = features_1[root_idx_new]
                features_short = torch.cat([features_short, root_extend], 1)

                features_short = F.relu(features_short)
                features_short = F.dropout(features_short, self.dropout, training=self.training)
                features_short = self.conv2(x=features_short, edge_index=adjs_short, edge_weight=values_short)
                features_short = F.relu(features_short)
                root_extend = torch.zeros(features_2.size(0), features_2.size(1)).to(self.device)
                for index in range(features_2.size(0)):
                    root_idx_new = root_idx[i] - node_num_count
                    root_extend[index] = features_2[root_idx_new]
                features_short = torch.cat([features_short, root_extend], 1)
                features_short = F.leaky_relu(self.linear(features_short))
                features_all = torch.cat([features_all,features_short],0)
                node_num_count += propagation_node_num[i].item()
                edge_num_count += propagation_edge_num[i].item()

        return features_all

class PropagationGraphConvLayer_nBatch_geometric(Module):
    def __init__(self, in_features,hidden_features, out_features,dropout,device, bias=True):
        super(PropagationGraphConvLayer_nBatch_geometric, self).__init__()
        self.conv1 = GCNConv(in_features,hidden_features,add_self_loops=False,normalize=False)
        self.conv2 = GCNConv(hidden_features+in_features, out_features, add_self_loops=False,normalize=False)
        # self.conv1 = GraphConvolution_nBatch(in_features, hidden_features)
        # self.conv2 = GraphConvolution_nBatch(hidden_features + in_features, out_features)
        self.linear = nn.Linear(hidden_features+out_features,in_features)
        self.dropout = dropout
        self.device = device
    def forward(self, features, adjs, values, root_idx, propagation_node_num, propagation_edge_num,batch):
        # print('feature',features.size())
        # print('----layer features----:', features.size())
        # print('----layer adjs----:', adjs.size())
        # print('----layer values----:', values.size())
        # print('----layer root_idx---:',root_idx)
        # print('----layer propagation node num---:',propagation_node_num)
        # print('----layer propagation edge num---:', propagation_edge_num)
        # print('----layer adjs----:', adjs)
        node_num_count = 0
        edge_num_count = 0
        # print('layers adjs:', adjs)
        # print('layers propagation_node_num:',propagation_node_num)
        # print('layers propagation_edge_num:',propagation_edge_num)
        # print('layers values:',values)
        features_1 = copy.copy(features.float())
        features = self.conv1(x=features,edge_index=adjs,edge_weight=values)
        # print('layers feature after conv1:',features)
        features_2 = copy.copy(features)
        root_extend = torch.zeros(len(batch), features_1.size(1)).to(self.device)
        # print('root_extend:',root_extend.size())
        # print('len features 1:',list(range(features_1.size(0))))
        # indices_slect_1 = torch.LongTensor(range(root_extend.size(0)))
        batch_size = max(batch)+1
        for num_batch in range(batch_size):
            index = (torch.eq(batch,num_batch))
            root_extend[index] = features_1[root_idx[num_batch]]
        # print('layers first root extend:',root_extend)
        features = torch.cat([features,root_extend],1)

        features = F.leaky_relu(features)
        #todo
        features = F.dropout(features,self.dropout,training=self.training)
        # print('layers feature after root extend:', features)
        features = self.conv2(x=features, edge_index=adjs,edge_weight=values)
        features = F.leaky_relu(features)
        root_extend = torch.zeros(len(batch),features_2.size(1)).to(self.device)
        for num_batch in range(batch_size):
            index = (torch.eq(batch,num_batch))
            root_extend[index] = features_2[root_idx[num_batch]]
        features = torch.cat([features,root_extend],1)
        features = F.leaky_relu(self.linear(features))
        # features_all = features_short
        #
        # node_num_count += propagation_node_num[i].item()
        # edge_num_count += propagation_edge_num[i].item()

        return features

#geometric knowledge GCN
class KnowledgeGraphConvLayer(Module):
    def __init__(self, in_features, hidden_features, out_features,dropout, bias=True):
        super(KnowledgeGraphConvLayer, self).__init__()
        self.conv1 = GraphConvolution(in_features,hidden_features)
        self.conv2 = GraphConvolution(hidden_features, out_features)
        self.dropout = dropout

    def forward(self, feature, adj):
        feature = self.conv1(feature,adj)
        feature = F.relu(feature)
        feature = F.dropout(feature,self.dropout, training=self.training)
        feature = self.conv2(feature,adj)
        return feature

class KnowledgeGraphConvLayer_nBatch(Module):
    def __init__(self, in_features, hidden_features, out_features,dropout,device, bias=True):
        super(KnowledgeGraphConvLayer_nBatch, self).__init__()
        # self.conv1 = GraphConvolution_nBatch(in_features,hidden_features,add_self_loops=False,normalize=False)
        # self.conv2 = GCNConv(hidden_features, out_features, add_self_loops=False,normalize=False)
        self.conv1 = GraphConvolution_nBatch(in_features,hidden_features)
        self.conv2 = GraphConvolution_nBatch(hidden_features, out_features)
        self.dropout = dropout
        self.device = device

    def forward(self, features,adjs,values,knowledge_node_num,knowledge_edge_num):
        node_num_count = 0
        edge_num_count = 0
        for i in range(len(knowledge_node_num)):
            if i == 0:
                indices_slect = torch.LongTensor(range(knowledge_edge_num[i].item())).to(self.device)
                features_short = features[0:knowledge_node_num[i].item()]
                adjs_short = torch.index_select(adjs,1,indices_slect).to(self.device)
                values_short = torch.index_select(values,0,indices_slect).to(self.device)
                # adjs_short = adjs[:][0:knowledge_edge_num[i].item()]
                # values_short = values[0:knowledge_edge_num[i].item()]
                # print('----layer features short  0----:', features_short.size())
                # print('----layer adjs short  0----:', adjs_short.size())
                # print('----layer adjs short----:', adjs_short)
                # print('----layer values short  0----:', values_short.size())
                # print('----layer edge num  0----:', knowledge_edge_num[i].item())
                # print('----layer node num  0----:', knowledge_node_num[i].item())
                features_short = self.conv1(x=features_short,edge_index=adjs_short,edge_weight=values_short)
                features_short = F.relu(features_short)
                features_short = F.dropout(features_short,self.dropout, training=self.training)
                features_short = self.conv2(x=features_short,edge_index=adjs_short,edge_weight=values_short)
                features_all = features_short
                node_num_count += knowledge_node_num[i].item()
                edge_num_count += knowledge_edge_num[i].item()
            else:
                # print('---------ok1--------')
                indices_slect = torch.LongTensor(range(edge_num_count,edge_num_count+knowledge_edge_num[i].item())).to(self.device)
                # print('---------ok2--------')
                features_short = features[node_num_count:node_num_count+knowledge_node_num[i].item()]
                # print('---------ok3--------')
                adjs_short = torch.index_select(adjs, 1, indices_slect).to(self.device)
                # print('---------ok4--------')
                values_short = torch.index_select(values, 0, indices_slect).to(self.device)
                # print('---------ok5--------')
                # adjs_short = adjs[:][edge_num_count:edge_num_count+knowledge_edge_num[i].item()]
                # values_short = values[edge_num_count:edge_num_count+knowledge_edge_num[i].item()]
                #TODO:没整明白
                adjs_copy = copy.copy(adjs_short)
                adjs_sub = torch.zeros(adjs_short.size(0),1).to(self.device)
                adjs_sub[0][0] = adjs_copy[0][0]
                adjs_sub[1][0] = adjs_copy[1][0]
                # print(id(adjs_short))
                # print(id(adjs_copy))
                # sub_num = adjs_copy[0][0]
                # print('sub_num',adjs_copy[0][0])
                # for j in range(adjs_short.size(1)):
                #     # print('sub_num 0',sub_num)
                #     adjs_short[0][j] = adjs_short[0][j]-adjs_[0][0]
                #     # print('sub_num 1', sub_num)
                #     adjs_short[1][j] = adjs_short[1][j]-adjs_copy[0][0]
                adjs_short = adjs_short - adjs_sub
                adjs_short = adjs_short.long()
                # print('sub_num after', adjs_copy[0][0])
                # print('----layer features short----:', features_short.size())
                # print('----layer adjs short----:', adjs_short.size())
                # print('----layer adjs short----:', adjs_short)
                # print('----layer values short----:', values_short.size())
                # print('----layer edge num----:', knowledge_edge_num[i].item())
                # print('----layer node num----:', knowledge_node_num[i].item())

                # features_short = self.conv1(x=features_short, edge_index=adjs_short, edge_weight=values_short)
                # features_short = F.relu(features_short)
                # features_short = F.dropout(features_short, self.dropout, training=self.training)
                # features_short = self.conv2(x=features_short, edge_index=adjs_short, edge_weight=values_short)
                features_all = torch.cat([features_all,features_short],0)
                node_num_count += knowledge_node_num[i].item()
                edge_num_count += knowledge_edge_num[i].item()
        return features_all

class KnowledgeGraphConvLayer_nBatch_geometric(Module):
    def __init__(self, in_features, hidden_features, out_features,dropout,device, bias=True):
        super(KnowledgeGraphConvLayer_nBatch_geometric, self).__init__()
        self.conv1 = GCNConv(in_features,hidden_features,add_self_loops=False,normalize=False)
        self.conv2 = GCNConv(hidden_features, out_features, add_self_loops=False,normalize=False)
        # self.conv1 = GraphConvolution_nBatch(in_features,hidden_features)
        # self.conv2 = GraphConvolution_nBatch(hidden_features, out_features)
        self.linear = nn.Linear(out_features,out_features)
        self.dropout = dropout
        self.device = device

    def forward(self, features,adjs,values,knowledge_node_num,knowledge_edge_num):


        features = self.conv1(x=features,edge_index=adjs,edge_weight=values)
        features = F.leaky_relu(features)
        #todo
        # features = F.dropout(features,self.dropout, training=self.training)
        features = self.conv2(x=features,edge_index=adjs,edge_weight=values)
        features = F.leaky_relu(self.linear(features))


        return features

class KnowledgeGraphConvLayer_nBatch_geometric_1(Module):
    def __init__(self, in_features, hidden_features, out_features,dropout,device, bias=True):
        super(KnowledgeGraphConvLayer_nBatch_geometric_1, self).__init__()
        self.conv1 = GCNConv(in_features,hidden_features,add_self_loops=False,normalize=False)
        self.conv2 = GCNConv(hidden_features+in_features, out_features, add_self_loops=False,normalize=False)
        # self.conv1 = GraphConvolution_nBatch(in_features,hidden_features)
        # self.conv2 = GraphConvolution_nBatch(hidden_features, out_features)
        self.linear = nn.Linear(out_features+hidden_features,in_features)
        self.dropout = dropout
        self.device = device

    def forward(self, features,adjs,values,root_idx,knowledge_node_num,knowledge_edge_num, batch):
        features_1 = copy.copy(features.float())
        features = self.conv1(x=features,edge_index=adjs,edge_weight=values)
        features_2 = copy.copy(features)

        batch_size = max(batch) + 1
        # print('batch_size', batch_size)
        graph_dim = torch.sum(knowledge_node_num)
        batch_k = torch.zeros(graph_dim).to(self.device)
        temp_num = 0
        for num_batch in range(batch_size):
            index = [idx for idx in range(temp_num, temp_num + knowledge_node_num[num_batch])]
            # print('index',index)
            temp_num += knowledge_node_num[num_batch]
            batch_k[index] = num_batch
        # print('batch_k', batch_k)
        assert len(batch_k) == features.size(0)

        root_extend = torch.zeros(len(batch_k),features_1.size(1)).to(self.device)
        for num_batch in range(batch_size):
            index = (torch.eq(batch_k,num_batch))
            root_extend[index] = features_1[root_idx[num_batch]]
        # print('root_extend',root_extend.size())
        features = torch.cat([features,root_extend],1)
        features = F.leaky_relu(features)
        # print('features', features.size())
        features = F.dropout(features, self.dropout, training=self.training)

        features = self.conv2(x=features,edge_index=adjs,edge_weight=values)
        features = F.leaky_relu(features)
        root_extend = torch.zeros(len(batch_k),features_2.size(1)).to(self.device)
        for num_batch in range(batch_size):
            index = (torch.eq(batch_k,num_batch))
            root_extend[index] = features_2[root_idx[num_batch]]
        features = torch.cat([features,root_extend],1)
        features = F.leaky_relu(self.linear(features))



        return features

#mask linear 没用
class MaskLinear(Module):
    def __init__(self):
        super(MaskLinear, self).__init__()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, in_features, out_features=1, bias=True):  # idx is a list
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.Tensor(in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        mask = torch.zeros(self.in_features).cuda()
        mask[idx] = x.squeeze()
        output = torch.matmul(self.weight, mask)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' => ' \
               + str(self.out_features) + ')'

#融合 Fusion
class TemporalEncoding(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(TemporalEncoding, self).__init__()
        out_features = int(in_features / 2)  # not useful now
        # out_o = out_c = int(in_features / 2)
        self.weight_o = Parameter(torch.Tensor(in_features, in_features))
        self.weight_p = Parameter(torch.Tensor(in_features, in_features))
        self.weight_k = Parameter(torch.Tensor(in_features, in_features))
        self.linear = nn.Linear(3*in_features, in_features)
        nn.init.xavier_uniform_(self.weight_o.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_p.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_k.data, gain=1.667)
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, h_p, h_k, mutual_index_p, mutual_index_k,h_o):
        trans_ho = torch.mm(h_o, self.weight_o)
        trans_hp = torch.mm(h_p, self.weight_p)
        trans_hk = torch.mm(h_k, self.weight_k)
        # output = torch.tanh((torch.cat((trans_ho, trans_hc), dim=1)))  # dim=1
        output = torch.tanh(torch.cat((trans_ho[mutual_index_p],trans_hp[mutual_index_p],trans_hk[mutual_index_k]),dim=1))
        # output_p = torch.zeros(h_p.size(0),h_p.size(1))
        # output_c = torch.zeros(h_k.)
        output = F.leaky_relu(self.linear(output))


        if self.bias is not None:
            output = output + self.bias

        trans_hp[mutual_index_p] = output
        trans_hk[mutual_index_k] = output
        return trans_hp,trans_hk

class TemporalEncoding_copy(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(TemporalEncoding_copy, self).__init__()
        out_features = int(in_features / 2)  # not useful now
        # out_o = out_c = int(in_features / 2)
        self.weight_o = Parameter(torch.Tensor(in_features, in_features))
        self.weight_c = Parameter(torch.Tensor(in_features, in_features))
        self.linear = nn.Linear(2*in_features, in_features)
        nn.init.xavier_uniform_(self.weight_o.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_c.data, gain=1.667)
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, h_p, h_k, mutual_index_p, mutual_index_k,last_x):
        trans_hp = torch.mm(h_p, self.weight_o)
        trans_hk = torch.mm(h_k, self.weight_c)
        # output = torch.tanh((torch.cat((trans_ho, trans_hc), dim=1)))  # dim=1
        output = torch.tanh(torch.cat((trans_hp[mutual_index_p],trans_hk[mutual_index_k]),dim=1))
        # output_p = torch.zeros(h_p.size(0),h_p.size(1))
        # output_c = torch.zeros(h_k.)
        output = F.leaky_relu(self.linear(output))


        if self.bias is not None:
            output = output + self.bias

        trans_hp[mutual_index_p] = output
        trans_hk[mutual_index_k] = output
        return trans_hp,trans_hk

class TemporalEncoding_nBatch(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(TemporalEncoding_nBatch, self).__init__()
        out_features = int(in_features / 2)  # not useful now
        # out_o = out_c = int(in_features / 2)
        self.weight_o = Parameter(torch.Tensor(in_features, in_features))
        self.weight_p = Parameter(torch.Tensor(in_features, in_features))
        self.weight_k = Parameter(torch.Tensor(in_features, in_features))
        self.linear = nn.Linear(3*in_features, in_features)
        nn.init.xavier_uniform_(self.weight_o.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_p.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_k.data, gain=1.667)
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, h_p, h_k, mutual_index_p, mutual_index_k,h_o,propagation_node_num,knowledge_node_num):
        trans_ho = torch.mm(h_o, self.weight_o)
        trans_hp = torch.mm(h_p, self.weight_p)
        trans_hk = torch.mm(h_k, self.weight_k)
        # output = torch.tanh((torch.cat((trans_ho, trans_hc), dim=1)))  # dim=1
        output = torch.tanh(torch.cat((trans_ho[mutual_index_p],trans_hp[mutual_index_p],trans_hk[mutual_index_k]),dim=1))
        # output_p = torch.zeros(h_p.size(0),h_p.size(1))
        # output_c = torch.zeros(h_k.)
        output = F.leaky_relu(self.linear(output))


        if self.bias is not None:
            output = output + self.bias

        trans_hp[mutual_index_p] = output
        trans_hk[mutual_index_k] = output
        return trans_hp,trans_hk

class TemporalEncoding_nBatch_geometric(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(TemporalEncoding_nBatch_geometric, self).__init__()
        out_features = int(in_features / 2)  # not useful now
        # out_o = out_c = int(in_features / 2)
        self.weight_o = Parameter(torch.Tensor(in_features, in_features))
        self.weight_p = Parameter(torch.Tensor(in_features, in_features))
        self.weight_k = Parameter(torch.Tensor(in_features, in_features))
        self.linear = nn.Linear(3*in_features, in_features)
        nn.init.xavier_uniform_(self.weight_o.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_p.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_k.data, gain=1.667)
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, h_p, h_k, mutual_index_p, mutual_index_k,h_o,propagation_node_num,knowledge_node_num):
        # print('gcn test layers h_o:',h_o[mutual_index_p])
        # print('gcn test layers h_p:',h_p[mutual_index_p])
        # print('gcn test layers h_k:', h_k[mutual_index_k])
        trans_ho = torch.mm(h_o[mutual_index_p], self.weight_o)
        trans_hp = torch.mm(h_p[mutual_index_p], self.weight_p)
        trans_hk = torch.mm(h_k[mutual_index_k], self.weight_k)
        # output = torch.tanh((torch.cat((trans_ho, trans_hc), dim=1)))  # dim=1
        output = torch.tanh(self.linear(torch.cat((trans_ho, trans_hp, trans_hk), dim=1)))
        # output_p = torch.zeros(h_p.size(0),h_p.size(1))
        # output_c = torch.zeros(h_k.)

        # output = F.leaky_relu(self.linear(output))

        if self.bias is not None:
            output = output + self.bias

        h_p[mutual_index_p] = output
        h_k[mutual_index_k] = output
        # print('gcn test layers h_p[mutual_index_p].size()',h_p[mutual_index_p].size())
        # print('gcn test layers h_k[mutual_index_k].size()', h_k[mutual_index_k].size())
        # print('gcn test layers output:',output)
        return h_p, h_k

class TemporalEncoding_nBatch_geometric_1(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(TemporalEncoding_nBatch_geometric_1, self).__init__()
        out_features = int(in_features / 2)  # not useful now
        # out_o = out_c = int(in_features / 2)
        self.weight_o = Parameter(torch.Tensor(in_features, in_features))
        self.weight_p = Parameter(torch.Tensor(in_features, in_features))
        self.weight_k = Parameter(torch.Tensor(in_features, in_features))
        self.linear = nn.Linear(3*in_features, in_features)
        # self.linear_2 = nn.Linear(2*in_features,in_features)
        nn.init.xavier_uniform_(self.weight_o.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_p.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_k.data, gain=1.667)
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, h_p, h_k, mutual_index_p, mutual_index_k,h_o,propagation_node_num,knowledge_node_num):
        # print('gcn test layers h_o:',h_o[mutual_index_p])
        # print('gcn test layers h_p:',h_p[mutual_index_p])
        # print('gcn test layers h_k:', h_k[mutual_index_k])
        trans_ho = torch.mm(h_o, self.weight_o)
        trans_hp = torch.mm(h_p, self.weight_p)
        trans_hk = torch.mm(h_k, self.weight_k)
        # output = torch.tanh((torch.cat((trans_ho, trans_hc), dim=1)))  # dim=1
        output = torch.tanh(self.linear(torch.cat((trans_ho[mutual_index_k], trans_hp[mutual_index_p], trans_hk[mutual_index_k]), dim=1)))
        # output_p = torch.zeros(h_p.size(0),h_p.size(1))
        # output_c = torch.zeros(h_k.)
        # new_hk = torch.tanh(self.linear_2(torch.cat((trans_ho,trans_hk),dim=1)))
        # output = F.leaky_relu(self.linear(output))

        if self.bias is not None:
            output = output + self.bias

        trans_hp[mutual_index_p] = output
        trans_hk[mutual_index_k] = output
        # print('gcn test layers h_p[mutual_index_p].size()',h_p[mutual_index_p].size())
        # print('gcn test layers h_k[mutual_index_k].size()', h_k[mutual_index_k].size())
        # print('gcn test layers output:',output)
        return trans_hp, trans_hk

class TemporalEncoding_nBatch_geometric_2(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(TemporalEncoding_nBatch_geometric_2, self).__init__()
        out_features = int(in_features / 2)  # not useful now
        # out_o = out_c = int(in_features / 2)
        self.weight_o = Parameter(torch.Tensor(in_features, in_features))
        self.weight_p = Parameter(torch.Tensor(in_features, in_features))
        self.weight_k = Parameter(torch.Tensor(in_features, in_features))
        self.linear = nn.Linear(3*in_features, in_features)
        self.linear_2 = nn.Linear(2*in_features,in_features)
        nn.init.xavier_uniform_(self.weight_o.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_p.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_k.data, gain=1.667)
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, h_p, h_k, mutual_index_p, mutual_index_k,h_o,propagation_node_num,knowledge_node_num):
        # print('gcn test layers h_o:',h_o[mutual_index_p])
        # print('gcn test layers h_p:',h_p[mutual_index_p])
        # print('gcn test layers h_k:', h_k[mutual_index_k])
        trans_ho = torch.mm(h_o, self.weight_o)
        trans_hp = torch.mm(h_p, self.weight_p)
        trans_hk = torch.mm(h_k, self.weight_k)
        # output = torch.tanh((torch.cat((trans_ho, trans_hc), dim=1)))  # dim=1
        output = torch.tanh(self.linear(torch.cat((trans_ho[mutual_index_k], trans_hp[mutual_index_p], trans_hk[mutual_index_k]), dim=1)))
        # output_p = torch.zeros(h_p.size(0),h_p.size(1))
        # output_c = torch.zeros(h_k.)
        new_hk = torch.tanh(self.linear_2(torch.cat((trans_ho,trans_hk),dim=1)))
        # output = F.leaky_relu(self.linear(output))

        if self.bias is not None:
            output = output + self.bias

        trans_hp[mutual_index_p] = output
        new_hk[mutual_index_k] = output
        # print('gcn test layers h_p[mutual_index_p].size()',h_p[mutual_index_p].size())
        # print('gcn test layers h_k[mutual_index_k].size()', h_k[mutual_index_k].size())
        # print('gcn test layers output:',output)
        return trans_hp, new_hk

class TemporalEncoding_nBatch_geometric_propagation(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(TemporalEncoding_nBatch_geometric_propagation, self).__init__()
        out_features = int(in_features / 2)  # not useful now
        # out_o = out_c = int(in_features / 2)
        self.weight_o = Parameter(torch.Tensor(in_features, in_features))
        self.weight_p = Parameter(torch.Tensor(in_features, in_features))
        self.weight_k = Parameter(torch.Tensor(in_features, in_features))
        self.linear = nn.Linear(2*in_features, in_features)
        # self.linear_2 = nn.Linear(2*in_features,in_features)
        nn.init.xavier_uniform_(self.weight_o.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_p.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_k.data, gain=1.667)
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, h_p, h_k, mutual_index_p, mutual_index_k,h_o,propagation_node_num,knowledge_node_num):
        # print('gcn test layers h_o:',h_o[mutual_index_p])
        # print('gcn test layers h_p:',h_p[mutual_index_p])
        # print('gcn test layers h_k:', h_k[mutual_index_k])
        trans_ho = torch.mm(h_o, self.weight_o)
        trans_hp = torch.mm(h_p, self.weight_p)
        # trans_hk = torch.mm(h_k, self.weight_k)
        # output = torch.tanh((torch.cat((trans_ho, trans_hc), dim=1)))  # dim=1
        output = torch.tanh(self.linear(torch.cat((trans_ho, trans_hp), dim=1)))
        # output_p = torch.zeros(h_p.size(0),h_p.size(1))
        # output_c = torch.zeros(h_k.)
        # new_hk = torch.tanh(self.linear_2(torch.cat((trans_ho,trans_hk),dim=1)))
        # output = F.leaky_relu(self.linear(output))

        if self.bias is not None:
            output = output + self.bias

        # trans_hp[mutual_index_p] = output
        # trans_hk[mutual_index_k] = output
        # print('gcn test layers h_p[mutual_index_p].size()',h_p[mutual_index_p].size())
        # print('gcn test layers h_k[mutual_index_k].size()', h_k[mutual_index_k].size())
        # print('gcn test layers output:',output)
        return output

class TemporalEncoding_nBatch_geometric_knowledge(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(TemporalEncoding_nBatch_geometric_knowledge, self).__init__()
        out_features = int(in_features / 2)  # not useful now
        # out_o = out_c = int(in_features / 2)
        self.weight_o = Parameter(torch.Tensor(in_features, in_features))
        self.weight_p = Parameter(torch.Tensor(in_features, in_features))
        self.weight_k = Parameter(torch.Tensor(in_features, in_features))
        self.linear = nn.Linear(2*in_features, in_features)
        # self.linear_2 = nn.Linear(2*in_features,in_features)
        nn.init.xavier_uniform_(self.weight_o.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_p.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_k.data, gain=1.667)
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, h_p, h_k, mutual_index_p, mutual_index_k,h_o,propagation_node_num,knowledge_node_num):
        # print('gcn test layers h_o:',h_o[mutual_index_p])
        # print('gcn test layers h_p:',h_p[mutual_index_p])
        # print('gcn test layers h_k:', h_k[mutual_index_k])
        trans_ho = torch.mm(h_o, self.weight_o)
        # trans_hp = torch.mm(h_p, self.weight_p)
        trans_hk = torch.mm(h_k, self.weight_k)
        # output = torch.tanh((torch.cat((trans_ho, trans_hc), dim=1)))  # dim=1
        output = torch.tanh(self.linear(torch.cat((trans_ho[mutual_index_k], trans_hk[mutual_index_k]), dim=1)))
        # output_p = torch.zeros(h_p.size(0),h_p.size(1))
        # output_c = torch.zeros(h_k.)
        # new_hk = torch.tanh(self.linear_2(torch.cat((trans_ho,trans_hk),dim=1)))
        # output = F.leaky_relu(self.linear(output))

        if self.bias is not None:
            output = output + self.bias

        # trans_hp[mutual_index_p] = output
        trans_hk[mutual_index_k] = output
        # print('gcn test layers h_p[mutual_index_p].size()',h_p[mutual_index_p].size())
        # print('gcn test layers h_k[mutual_index_k].size()', h_k[mutual_index_k].size())
        # print('gcn test layers output:',output)
        return output,trans_hk

class TemporalEncoding_nBatch_geometric_wotf(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(TemporalEncoding_nBatch_geometric_wotf, self).__init__()
        out_features = int(in_features / 2)  # not useful now
        # out_o = out_c = int(in_features / 2)
        self.weight_o = Parameter(torch.Tensor(in_features, in_features))
        self.weight_p = Parameter(torch.Tensor(in_features, in_features))
        self.weight_k = Parameter(torch.Tensor(in_features, in_features))
        self.linear = nn.Linear(2*in_features, in_features)
        # self.linear_2 = nn.Linear(2*in_features,in_features)
        nn.init.xavier_uniform_(self.weight_o.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_p.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_k.data, gain=1.667)
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, h_p, h_k, mutual_index_p, mutual_index_k,h_o,propagation_node_num,knowledge_node_num):
        # print('gcn test layers h_o:',h_o[mutual_index_p])
        # print('gcn test layers h_p:',h_p[mutual_index_p])
        # print('gcn test layers h_k:', h_k[mutual_index_k])
        # trans_ho = torch.mm(h_o, self.weight_o)
        # trans_hp = torch.mm(h_p, self.weight_p)
        # trans_hk = torch.mm(h_k, self.weight_k)
        # output = torch.tanh((torch.cat((trans_ho, trans_hc), dim=1)))  # dim=1
        output = torch.tanh(self.linear(torch.cat((h_p[mutual_index_p], h_k[mutual_index_k]), dim=1)))
        # output_p = torch.zeros(h_p.size(0),h_p.size(1))
        # output_c = torch.zeros(h_k.)
        # new_hk = torch.tanh(self.linear_2(torch.cat((trans_ho,trans_hk),dim=1)))
        # output = F.leaky_relu(self.linear(output))

        if self.bias is not None:
            output = output + self.bias

        h_p[mutual_index_p] = output
        h_k[mutual_index_k] = output
        # print('gcn test layers h_p[mutual_index_p].size()',h_p[mutual_index_p].size())
        # print('gcn test layers h_k[mutual_index_k].size()', h_k[mutual_index_k].size())
        # print('gcn test layers output:',output)
        return h_p, h_k

class TemporalEncoding_nBatch_geometric_attention(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(TemporalEncoding_nBatch_geometric_attention, self).__init__()
        out_features = int(in_features / 2)  # not useful now
        # out_o = out_c = int(in_features / 2)
        self.weight_o = Parameter(torch.Tensor(in_features, out_features))
        self.weight_p = Parameter(torch.Tensor(in_features, out_features))
        self.weight_k = Parameter(torch.Tensor(in_features, out_features))
        self.linear = nn.Linear(in_features, in_features)
        nn.init.xavier_uniform_(self.weight_o.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_p.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_k.data, gain=1.667)
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

        #多头attention
        self.num_head = 1
        self.dim_model = 128
        assert self.dim_model % self.num_head == 0
        self.dim_head = self.dim_model // self.num_head
        self.fc_Q = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_K = nn.Linear(self.dim_model, self.num_head * self.dim_head)
        self.fc_V = nn.Linear(self.dim_model, self.num_head * self.dim_head)

        self.attention = Scaled_Dot_Product_Attention()
        self.fc1 = nn.Linear(self.num_head * self.dim_head, self.dim_model)
        # self.fc2 = nn.Linear(self.num_head * self.dim_head, self.dim_model)
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(self.dim_model)

    def forward(self, h_p, h_k, mutual_index_p, mutual_index_k,h_o,propagation_node_num,knowledge_node_num):
        temp_batch_size = h_o.size()[0]
        Q = h_o[mutual_index_p].unsqueeze(1) #200,1,128
        K_p = h_p[mutual_index_p].unsqueeze(1)
        K_k = h_k[mutual_index_k].unsqueeze(1)

        K = torch.cat([K_p,K_k],dim=1)
        print('layer attention K size:',K.size()) #200,2,128

        Q = self.fc_Q(Q)
        K = self.fc_K(K)
        V = self.fc_V(K)
        Q = Q.view(temp_batch_size*self.num_head,-1,self.dim_head)
        K = K.view(temp_batch_size * self.num_head, -1, self.dim_head)
        V = V.view(temp_batch_size * self.num_head, -1, self.dim_head)

        scale = K.size(-1) ** -0.5
        fusion_pk = self.attention(Q,K,V,scale)
        fusion_pk = fusion_pk.view(temp_batch_size,-1,self.dim_head*self.num_head)
        fusion_pk = self.fc1(fusion_pk)
        fusion_pk = self.dropout(fusion_pk)
        fusion_pk = self.layer_norm(fusion_pk)
        print('layer attention fusion size:', fusion_pk.size())

        fusion_pk = fusion_pk.squeeze(1)

        trans_ho = torch.mm(h_o[mutual_index_p], self.weight_o)
        trans_pk = torch.mm(fusion_pk, self.weight_p)
        output = torch.tanh(torch.cat((trans_ho,trans_pk),dim=1))
        # output_p = torch.zeros(h_p.size(0),h_p.size(1))
        # output_c = torch.zeros(h_k.)
        output = F.leaky_relu(self.linear(output))


        if self.bias is not None:
            output = output + self.bias

        h_p[mutual_index_p] = output
        h_k[mutual_index_k] = output
        return h_p,h_k

class TemporalEncoding_nBatch_geometric_static(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(TemporalEncoding_nBatch_geometric_static, self).__init__()
        out_features = int(in_features / 2)  # not useful now
        # out_o = out_c = int(in_features / 2)
        self.weight_o = Parameter(torch.Tensor(in_features, in_features))
        self.weight_p = Parameter(torch.Tensor(in_features, in_features))
        self.weight_k = Parameter(torch.Tensor(in_features, in_features))
        self.linear = nn.Linear(3*in_features, in_features)
        nn.init.xavier_uniform_(self.weight_o.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_p.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_k.data, gain=1.667)
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, h_p, h_k, mutual_index_p, mutual_index_k,h_o,propagation_node_num,knowledge_node_num):
        trans_ho = torch.mm(h_o[mutual_index_p], self.weight_o)
        trans_hp = torch.mm(h_p[mutual_index_p], self.weight_p)
        trans_hk = torch.mm(h_k[mutual_index_k], self.weight_k)
        # output = torch.tanh((torch.cat((trans_ho, trans_hc), dim=1)))  # dim=1
        output = torch.tanh(torch.cat((trans_ho,trans_hp,trans_hk),dim=1))
        # output_p = torch.zeros(h_p.size(0),h_p.size(1))
        # output_c = torch.zeros(h_k.)
        output = F.leaky_relu(self.linear(output))


        if self.bias is not None:
            output = output + self.bias

        h_p[mutual_index_p] = output
        h_k[mutual_index_k] = output
        return h_p,h_k

class TemporalEncoding_nBatch_static_knowledge(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(TemporalEncoding_nBatch_static_knowledge, self).__init__()
        out_features = int(in_features / 2)  # not useful now
        # out_o = out_c = int(in_features / 2)
        self.weight_o = Parameter(torch.Tensor(in_features, in_features))
        # self.weight_p = Parameter(torch.Tensor(in_features, in_features))
        self.weight_k = Parameter(torch.Tensor(in_features, in_features))
        self.linear = nn.Linear(2*in_features, in_features)
        nn.init.xavier_uniform_(self.weight_o.data, gain=1.667)
        # nn.init.xavier_uniform_(self.weight_p.data, gain=1.667)
        nn.init.xavier_uniform_(self.weight_k.data, gain=1.667)
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, h_k, mutual_index_p, mutual_index_k,h_o,propagation_node_num,knowledge_node_num):
        trans_ho = torch.mm(h_o, self.weight_o)
        # trans_hp = torch.mm(h_p, self.weight_p)
        trans_hk = torch.mm(h_k, self.weight_k)
        # output = torch.tanh((torch.cat((trans_ho, trans_hc), dim=1)))  # dim=1
        output = torch.tanh(torch.cat((trans_ho[mutual_index_p],trans_hk[mutual_index_k]),dim=1))
        # output_p = torch.zeros(h_p.size(0),h_p.size(1))
        # output_c = torch.zeros(h_k.)
        output = F.leaky_relu(self.linear(output))


        if self.bias is not None:
            output = output + self.bias

        # trans_hp[mutual_index_p] = output
        trans_hk[mutual_index_k] = output
        return output,trans_hk


#mean
class Mean_nBatch(Module):
    def __init__(self,device,n_feature,n_output):
        super(Mean_nBatch, self).__init__()
        self.device = device
        self.linear = nn.Linear(n_feature, n_output)

    def forward(self,x,node_num):
        node_num_count = 0
        for i in range(len(node_num)):
            if i == 0:
                indices_slect = torch.LongTensor(range(node_num[i].item())).to(self.device)
                # features_short = features[0:knowledge_node_num[i].item()]
                # adjs_short = torch.index_select(adjs, 1, indices_slect).to(self.device)
                # x_short = x[0:node_num[i].item()]
                x_short = torch.index_select(x, 0, indices_slect).to(self.device)
                # print('----layer x short----',x_short.size())
                x_short = torch.mean(x_short, dim=0)
                # print('----layer x short after mean----', x_short.size())
                x_short = x_short.unsqueeze(0)
                x_short = self.linear(x_short)
                x_short = torch.sigmoid(x_short)
                x_all = x_short
                node_num_count += node_num[i].item()
            else:
                indices_slect = torch.LongTensor(range(node_num_count,node_num_count+node_num[i].item())).to(self.device)
                # x_short = x[node_num_count:node_num_count+node_num[i].item()]
                x_short = torch.index_select(x, 0, indices_slect).to(self.device)
                x_short = torch.mean(x_short,dim=0)
                x_short = x_short.unsqueeze(0)
                x_short = self.linear(x_short)
                x_short = torch.sigmoid(x_short)
                x_all = torch.cat([x_all,x_short],0)
                node_num_count += node_num[i].item()
        return x_all

class Mean_nBatch_geometric(Module):
    def __init__(self,device,n_feature,n_output):
        super(Mean_nBatch_geometric, self).__init__()
        self.device = device
        self.linear = nn.Linear(n_feature, n_output)

    def forward(self,x,batch):
        x = scatter_mean(x,batch,dim=0)

        return x


#gru
class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUEncoder, self).__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        # self.text_length = text_length

        # bigru encoder
        self.gru = nn.GRU(input_dim, hidden_dim, bidirectional=False)
        # self.hidden = self.init_hidden()


    def init_hidden(self):
        return nn.Parameter(torch.zeros(2, 1, self.hidden_dim))

    def forward(self, text_embeddings, text_lengths):
        # text_embeddings = torch.nn.utils.rnn.pack_padded_sequence(text_embeddings, text_lengths, batch_first=True,enforce_sorted=False)
        gru_out, _ = self.gru(text_embeddings)        # [seq_len, batch_size, 2 * hidden_dim]
        # gru_out = torch.nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True, padding_value=0.0, total_length=None)[0]
        gru_out = gru_out[-1, :, :]
        return gru_out

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BiLSTMEncoder, self).__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        # self.text_length = text_length

        # bigru encoder
        self.gru = nn.LSTM(input_dim, hidden_dim, num_layers=1, bias=True, batch_first=True, bidirectional=True)
        # self.hidden = self.init_hidden()


    def init_hidden(self):
        return nn.Parameter(torch.zeros(2, 1, self.hidden_dim))

    def forward(self, text_embeddings, text_lengths):
        # text_embeddings = torch.nn.utils.rnn.pack_padded_sequence(text_embeddings, text_lengths, batch_first=True,enforce_sorted=False)
        gru_out, _ = self.gru(text_embeddings)        # [seq_len, batch_size, 2 * hidden_dim]
        # gru_out = torch.nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True, padding_value=0.0, total_length=None)[0]
        gru_out = gru_out[0, -1, :]
        return gru_out

class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        # print('----attention------------:',attention)
        context = torch.matmul(attention, V)
        return context
