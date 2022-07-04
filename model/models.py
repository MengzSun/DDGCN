#data-flow p_node -->GRU--> GCN-->fusioned
#          k_node       -->GCN -->fusioned
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
# from torch_sparse import spmm
from layers import *
from tqdm import tqdm
from transformers import BertModel, BertConfig

class dual_DynamicGCN_nBatch_copy(nn.Module):
    def __init__(self, token_dict,mid2regtextidx_dict,n_output,config,device, n_hidden=3, dropout=0.2, instance_norm=False):
        super(dual_DynamicGCN_nBatch_copy, self).__init__()
        self.token_dict = token_dict
        self.mid2regtextidx_dict = mid2regtextidx_dict
        self.config = config
        self.dropout = dropout
        self.instance_norm = instance_norm
        if self.instance_norm:
            self.norm = nn.InstanceNorm1d(self.config.graph_embedding_dim, momentum=0.0, affine=True)
        self.n_output = n_output
        self.device = device

        #text
        self.embedding = nn.Embedding(len(self.token_dict), config.graph_embedding_dim)
        self.gru = GRUEncoder(self.config.graph_embedding_dim, self.config.hidden_dim)
        # self.embedding.weight = nn.Parameter(pretrained_emb)
        # self.embedding.weight.requires_grad = False

        self.n_feature = self.config.hidden_dim
        # self.n_input = pretrained_emb.size(0)

        self.layer_stack_propagation = nn.ModuleList()  # TODO class initiate
        self.layer_stack_knowledge = nn.ModuleList()
        self.bn_stack_propagation = nn.ModuleList()
        self.bn_stack_knowledge = nn.ModuleList()
        self.temporal_cells = nn.ModuleList()
        for i in range(n_hidden):
            self.layer_stack_propagation.append(PropagationGraphConvLayer_nBatch(self.n_feature, self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_propagation.append(nn.BatchNorm1d(self.n_feature))
            self.layer_stack_knowledge.append(KnowledgeGraphConvLayer_nBatch(self.n_feature,self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_knowledge.append(nn.BatchNorm1d(self.n_feature))
            self.temporal_cells.append(TemporalEncoding_nBatch(self.n_feature, self.n_feature))

        # self.layer_stack_propagation.append(SpGraphConvLayer(self.n_feature, self.n_feature))
        # self.layer_stack_knowledge.append(SpGraphConvLayer(self.n_feature,self.n_feature))
        # self.bn_stack.append(nn.BatchNorm1d(2*self.n_feature))
        self.mean = Mean_nBatch(self.device,self.n_feature,self.n_output)
        # self.linear = nn.Linear(self.n_feature,self.n_output)
        # self.mask = MaskLinear()
        self.save_x = None
        self.text_length = torch.tensor([self.config.text_max_length]).to(self.device)

    def forward(self, data):
        idx2encoding = {}
        features_propagation = []
        features_knowledge = []
        # for mid in self.mid2regtextidx_dict:
        #     # if mid ==
        #     print('-----mid',type(mid)) #str
            # print('-----textidx',self.mid2regtextidx_dict[mid])
        idx_propagation_list = data.x_propagation_idx.cpu().numpy().tolist()
        idx_knowledge_list = data.x_knowledge_idx.cpu().numpy().tolist()
        # print('idx_propagation_list:',idx_propagation_list)

        # print('data root idx:',data.root_idx, type(data.root_idx), data.root_idx.size())
        root_idx_p = []
        for root_idx in data.root_idx:
            root_idx_item = root_idx.item()
            root_idx_p.append(idx_propagation_list.index(root_idx_item))
        root_idx_p = torch.tensor(root_idx_p)
        # print('---model root_idx_p---:',root_idx_p)
        # print('---model node num----:',data.x_propagation_node_num)
        for idx in data.x_propagation_idx:
            # if idx in self.token_dict:
            #     vertices_propagation.append(self.token_dict[idx])
            # elif idx in self.mid2regtextidx_dict:
            # print('--------:',idx.item())

            idx = idx.item()
            # print('model bug len(self.token_dict):',len(self.token_dict))
            # print('model bug self.mid2regtextidx_dict[idx]:', self.mid2regtextidx_dict[idx])
            text_embedding = self.embedding(torch.LongTensor(self.mid2regtextidx_dict[idx]).to(self.device))
            text_embedding = torch.unsqueeze(text_embedding,dim=1)
            # print('---text embedding---:',text_embedding.size()) #120,1,128

            text_encoding = self.gru(text_embedding,self.text_length) #120,1,256
            # print('text encoding:',text_encoding.size())
            idx2encoding[idx] = text_encoding
            features_propagation.append(text_encoding.detach().cpu().numpy())
        # for idx in
        dual_graph_mutual_index_p = []
        dual_graph_mutual_index_k = []
        for idx in data.x_knowledge_idx:
            idx = idx.item()
            if idx in self.token_dict:
                # print('idx:',idx,'token dict idx:',self.token_dict[idx],'token dict type:',type(self.token_dict[idx]))
                token_idx = torch.LongTensor([self.token_dict[idx]]).to(self.device)
                # print('token idx',token_idx)
                entity_embedding = self.embedding(token_idx)
                # print('entity embedding:',entity_embedding.size()) #1,128
                # print('embedding',entity_embedding)
                # except RuntimeError:
                # print('error idx:',idx)
                # else:
                features_knowledge.append(entity_embedding.detach().cpu().numpy())
            elif idx in self.mid2regtextidx_dict:
                dual_graph_mutual_index_p.append(idx_propagation_list.index(idx))
                dual_graph_mutual_index_k.append(idx_knowledge_list.index(idx))
                features_knowledge.append(idx2encoding[idx].detach().cpu().numpy())
        # print('model dual_graph_mutual_index_p:',dual_graph_mutual_index_p)
        # print('model dual_graph_mutual_index_k:',dual_graph_mutual_index_k)
        features_propagation = torch.tensor(features_propagation)
        features_knowledge = torch.tensor(features_knowledge)

        # emb_propagation = self.embedding(vertices_propagation)
        if self.instance_norm:
            features_propagation = self.norm(features_propagation)

        # emb_knowledge = self.embedding(vertices_knowledge)
        if self.instance_norm:
            features_knowledge = self.norm(features_knowledge)

        x_propagation = features_propagation
        x_knowledge = features_knowledge
        # print('x_propagation:',features_propagation.size()) #7,1,128
        # print('x_knowleddge:',features_knowledge.size()) #771,1,128
        x_propagation = torch.squeeze(x_propagation,dim=1)
        x_knowledge = torch.squeeze(x_knowledge,dim=1)
        # print('x_propagation:', x_propagation.size())  # 7,1,128
        # print('x_knowleddge:', x_knowledge.size())  # 771,1,128
        # print('adj_propagation:',adjs_propagation[0].size())
        # print('adj_knowledge:',adjs_knowledge[0].size())

        x_propagation = x_propagation.to(self.device)
        x_knowledge = x_knowledge.to(self.device)
        x_propagation_node_indices = [data.edge_index,data.x_propagation_node_indices_1_edge_index,data.x_propagation_node_indices_2_edge_index]
        x_propagation_node_values = [data.x_propagation_node_values_0,data.x_propagation_node_values_1,data.x_propagation_node_values_2]
        x_progation_edge_num = [data.x_propagation_edge_num_0,data.x_propagation_edge_num_1,data.x_propagation_edge_num_2]

        x_knowledge_node_indices = [data.x_knowledge_node_indices_0_edge_index, data.x_knowledge_node_indices_1_edge_index,
                                      data.x_knowledge_node_indices_2_edge_index]
        x_knowledge_node_values = [data.x_knowledge_node_values_0, data.x_knowledge_node_values_1,
                                     data.x_knowledge_node_values_2]
        x_knowledge_edge_num = [data.x_knowledge_edge_num_0, data.x_knowledge_edge_num_1,
                                data.x_knowledge_edge_num_2]
        # r = []
        # TODO: dynamic_graph那一篇paper 代码和文中所提不太一样
        last_x = x_propagation
        for i, gcn_layer in enumerate(self.layer_stack_propagation):
            # last_x = x
            #indices, values, size to torch.sparse matrix
            # print('x_propagation_node_indices',data.x_propagation_node_indices[i], len(data.x_propagation_node_indices))
            # shape_matrix = torch.zeros(data.x_propagation_node_indices[i].size(1),data.x_propagation_node_indices[i].size(1))
            # shape = torch.Size(shape_matrix)
            # x_propagation_node_sparse = torch.sparse.FloatTensor(data.x_propagation_node_indices[i], data.x_propagation_node_values, shape)
            x_gcn = gcn_layer(x_propagation, x_propagation_node_indices[i], x_propagation_node_values[i], root_idx_p, data.x_propagation_node_num, x_progation_edge_num[i])
            # print('x_gcn:',x_gcn.size())
            x_propagation = self.bn_stack_propagation[i](x_gcn)
            x_knowledge = self.bn_stack_knowledge[i](self.layer_stack_knowledge[i](x_knowledge,x_knowledge_node_indices[i],x_knowledge_node_values[i],data.x_knowledge_node_num,x_knowledge_edge_num[i]))
            # if i < len(self.layer_stack) - 1:
            x_propagation = F.relu(x_propagation)
            x_knowledge = F.relu(x_knowledge)

            x_propagation = F.dropout(x_propagation, self.dropout, training=self.training)
            x_knowledge = F.dropout(x_knowledge,self.dropout,training = self.training)

            x_propagation, x_knowledge = self.temporal_cells[i](x_propagation, x_knowledge,dual_graph_mutual_index_p,dual_graph_mutual_index_k,last_x,data.x_propagation_node_num,data.x_knowledge_node_num)  # temporal encoding

        x_propagation = F.relu(x_propagation)
        # x_knowledge = F.relu(x_knowledge)
        # x = torch.cat([x_propagation,x_knowledge],1)
        # print('---model x propagation---:', x_propagation.size())
        x = self.mean(x_propagation, data.x_propagation_node_num)
        x = x.squeeze(1)
        # print('---model x ---:', x)
        # x = self.linear(x)
        # x = self.mask(x, vertices)
        # print('----result before sigmoid----:',x)
        # x = torch.sigmoid(x)
        # print('----result after sigmoid----:',x)
        return x

class dual_DynamicGCN_nBatch(nn.Module):
    def __init__(self, token_dict,mid2regtextidx_dict,n_output,config,device, n_hidden=3, dropout=0.2, instance_norm=False):
        super(dual_DynamicGCN_nBatch, self).__init__()
        self.token_dict = token_dict
        self.mid2regtextidx_dict = mid2regtextidx_dict
        self.config = config
        self.dropout = dropout
        self.instance_norm = instance_norm
        if self.instance_norm:
            self.norm = nn.InstanceNorm1d(self.config.graph_embedding_dim, momentum=0.0, affine=True)
        self.n_output = n_output
        self.device = device

        #text
        self.embedding = nn.Embedding(len(self.token_dict), config.graph_embedding_dim)
        self.gru = GRUEncoder(self.config.graph_embedding_dim, self.config.hidden_dim)
        # self.embedding.weight = nn.Parameter(pretrained_emb)
        # self.embedding.weight.requires_grad = False

        self.n_feature = self.config.hidden_dim
        # self.n_input = pretrained_emb.size(0)

        self.layer_stack_propagation = nn.ModuleList()  # TODO class initiate
        self.layer_stack_knowledge = nn.ModuleList()
        self.bn_stack_propagation = nn.ModuleList()
        self.bn_stack_knowledge = nn.ModuleList()
        self.temporal_cells = nn.ModuleList()
        for i in range(n_hidden):
            self.layer_stack_propagation.append(PropagationGraphConvLayer_nBatch_geometric(self.n_feature, self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_propagation.append(nn.BatchNorm1d(self.n_feature))
            self.layer_stack_knowledge.append(KnowledgeGraphConvLayer_nBatch_geometric(self.n_feature,self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_knowledge.append(nn.BatchNorm1d(self.n_feature))
            self.temporal_cells.append(TemporalEncoding_nBatch_geometric(self.n_feature, self.n_feature))

        # self.layer_stack_propagation.append(SpGraphConvLayer(self.n_feature, self.n_feature))
        # self.layer_stack_knowledge.append(SpGraphConvLayer(self.n_feature,self.n_feature))
        # self.bn_stack.append(nn.BatchNorm1d(2*self.n_feature))
        self.mean = Mean_nBatch_geometric(self.device,self.n_feature,self.n_output)
        self.linear = nn.Linear(self.n_feature,self.n_output)
        # self.mask = MaskLinear()
        self.save_x = None
        self.text_length = torch.tensor([self.config.text_max_length]).to(self.device)

    def forward(self, data):
        idx2encoding = {}
        features_propagation = []
        features_knowledge = []

        #将图节点tensor转换为list
        idx_propagation_list = data.x.squeeze(1).cpu().numpy().tolist()
        idx_knowledge_list = data.x_knowledge_idx.squeeze(1).cpu().numpy().tolist()
        # print('---model idx_propagation_list---:', idx_propagation_list)

        #将root_idx对应node2idx转换为在图中的
        root_idx_p = []
        for root_idx in data.root_idx:
            root_idx_item = root_idx.item()
            root_idx_p.append(idx_propagation_list.index(root_idx_item))
        root_idx_p = torch.tensor(root_idx_p)
        # print('---model root_idx_p---:',root_idx_p)


        for idx in data.x:
            # if idx in self.token_dict:
            #     vertices_propagation.append(self.token_dict[idx])
            # elif idx in self.mid2regtextidx_dict:
            # print('--------:',idx.item())

            idx = idx.item()
            # print('model bug len(self.token_dict):',len(self.token_dict))
            # print('model bug self.mid2regtextidx_dict[idx]:', self.mid2regtextidx_dict[idx])
            text_embedding = self.embedding(torch.LongTensor(self.mid2regtextidx_dict[idx]).to(self.device))
            text_embedding = torch.unsqueeze(text_embedding,dim=1)
            # print('---text embedding---:',text_embedding.size()) #120,1,128
            text_encoding = torch.mean(text_embedding,dim=0)
            # print('model text encoding:',text_encoding.size())
            #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            #weibo数据集gru这部分总是报错，如果去掉gru呢
            # text_encoding = self.gru(text_embedding,self.text_length) #120,1,256
            # print('models text encoding:',text_encoding)
            #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            idx2encoding[idx] = text_encoding
            features_propagation.append(text_encoding.detach().cpu().numpy())
        # for idx in
        dual_graph_mutual_index_p = []
        dual_graph_mutual_index_k = []
        for idx in data.x_knowledge_idx:
            idx = idx.item()
            if idx in self.token_dict:
                # print('idx:',idx,'token dict idx:',self.token_dict[idx],'token dict type:',type(self.token_dict[idx]))
                token_idx = torch.LongTensor([self.token_dict[idx]]).to(self.device)
                # print('token idx',token_idx)
                entity_embedding = self.embedding(token_idx)
                # print('entity embedding:',entity_embedding.size()) #1,128
                # print('embedding',entity_embedding)
                # except RuntimeError:
                # print('error idx:',idx)
                # else:
                features_knowledge.append(entity_embedding.detach().cpu().numpy())
            elif idx in self.mid2regtextidx_dict:
                dual_graph_mutual_index_p.append(idx_propagation_list.index(idx))
                dual_graph_mutual_index_k.append(idx_knowledge_list.index(idx))
                features_knowledge.append(idx2encoding[idx].detach().cpu().numpy())
        # print('model dual_graph_mutual_index_p:',dual_graph_mutual_index_p)
        # print('model dual_graph_mutual_index_k:',dual_graph_mutual_index_k)
        features_propagation = torch.tensor(features_propagation)
        features_knowledge = torch.tensor(features_knowledge)

        # emb_propagation = self.embedding(vertices_propagation)
        if self.instance_norm:
            features_propagation = self.norm(features_propagation)

        # emb_knowledge = self.embedding(vertices_knowledge)
        if self.instance_norm:
            features_knowledge = self.norm(features_knowledge)

        x_propagation = features_propagation
        x_knowledge = features_knowledge
        # print('x_propagation:',features_propagation.size()) #7,1,128
        # print('x_knowleddge:',features_knowledge.size()) #771,1,128
        x_propagation = torch.squeeze(x_propagation,dim=1)
        x_knowledge = torch.squeeze(x_knowledge,dim=1)
        # print('x_propagation:', x_propagation.size())  # 7,1,128
        # print('x_knowleddge:', x_knowledge.size())  # 771,1,128
        # print('adj_propagation:',adjs_propagation[0].size())
        # print('adj_knowledge:',adjs_knowledge[0].size())

        x_propagation = x_propagation.to(self.device)
        x_knowledge = x_knowledge.to(self.device)
        x_propagation_node_indices = [data.edge_index,data.x_propagation_node_indices_1_edge_index,data.x_propagation_node_indices_2_edge_index]
        x_propagation_node_values = [data.edge_values,data.x_propagation_node_values_1,data.x_propagation_node_values_2]
        x_progation_edge_num = [data.edge_num,data.x_propagation_edge_num_1,data.x_propagation_edge_num_2]

        x_knowledge_node_indices = [data.x_knowledge_node_indices_0_edge_index, data.x_knowledge_node_indices_1_edge_index,
                                      data.x_knowledge_node_indices_2_edge_index]
        x_knowledge_node_values = [data.x_knowledge_node_values_0, data.x_knowledge_node_values_1,
                                     data.x_knowledge_node_values_2]
        x_knowledge_edge_num = [data.x_knowledge_edge_num_0, data.x_knowledge_edge_num_1,
                                data.x_knowledge_edge_num_2]
        # r = []
        # TODO: dynamic_graph那一篇paper 代码和文中所提不太一样
        last_x = x_propagation
        for i, gcn_layer in enumerate(self.layer_stack_propagation):
            # last_x = x
            #indices, values, size to torch.sparse matrix
            # print('x_propagation_node_indices',data.x_propagation_node_indices[i], len(data.x_propagation_node_indices))
            # shape_matrix = torch.zeros(data.x_propagation_node_indices[i].size(1),data.x_propagation_node_indices[i].size(1))
            # shape = torch.Size(shape_matrix)
            # x_propagation_node_sparse = torch.sparse.FloatTensor(data.x_propagation_node_indices[i], data.x_propagation_node_values, shape)
            x_gcn = gcn_layer(x_propagation, x_propagation_node_indices[i], x_propagation_node_values[i], root_idx_p, data.x_propagation_node_num, x_progation_edge_num[i],data.batch)
            # print('x_gcn:',x_gcn.size())
            x_propagation = self.bn_stack_propagation[i](x_gcn)
            x_knowledge = self.bn_stack_knowledge[i](self.layer_stack_knowledge[i](x_knowledge,x_knowledge_node_indices[i],x_knowledge_node_values[i],data.x_knowledge_node_num,x_knowledge_edge_num[i]))
            # if i < len(self.layer_stack) - 1:
            x_propagation = F.relu(x_propagation)
            x_knowledge = F.relu(x_knowledge)

            x_propagation = F.dropout(x_propagation, self.dropout, training=self.training)
            x_knowledge = F.dropout(x_knowledge,self.dropout,training = self.training)

            x_propagation, x_knowledge = self.temporal_cells[i](x_propagation, x_knowledge,dual_graph_mutual_index_p,dual_graph_mutual_index_k,last_x,data.x_propagation_node_num,data.x_knowledge_node_num)  # temporal encoding

        x_propagation = F.relu(x_propagation)
        # x_knowledge = F.relu(x_knowledge)
        # x = torch.cat([x_propagation,x_knowledge],1)
        # print('---model x propagation---:', x_propagation.size())
        x = self.mean(x_propagation, data.batch)
        x = x.squeeze(1)
        # print('---model x ---:', x)
        x = self.linear(x)
        # x = self.mask(x, vertices)
        # print('----result before sigmoid----:',x)
        x = torch.sigmoid(x)
        # print('----result after sigmoid----:',x,x.size())
        return x

#========================================================================================

#只有动态传播图的模型
class dual_DynamicGCN_nBatch_propagation(nn.Module):
    def __init__(self, token_dict, mid2bert_tokenizer, bert_path, n_output,config,device, n_hidden=3, dropout=0.2, instance_norm=False):
        super(dual_DynamicGCN_nBatch_propagation, self).__init__()
        self.token_dict = token_dict
        self.mid2bert_tokenizer = mid2bert_tokenizer
        self.config = config
        self.dropout = dropout
        self.instance_norm = instance_norm
        if self.instance_norm:
            self.norm = nn.InstanceNorm1d(self.config.graph_embedding_dim, momentum=0.0, affine=True)
        self.n_output = n_output
        self.device = device
        self.text_length = torch.tensor([self.config.text_max_length]).to(self.device)

        #-----------------------text-----------------------------------
        self.embedding = nn.Embedding(len(self.token_dict), config.graph_embedding_dim)
        self.gru = GRUEncoder(self.config.graph_embedding_dim, self.config.hidden_dim)
        # self.embedding.weight = nn.Parameter(pretrained_emb)
        # self.embedding.weight.requires_grad = False

        self.n_feature = self.config.hidden_dim
        # self.n_input = pretrained_emb.size(0)

        #------------------ ----bert-----------------------------------
        modelConfig = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path, config=modelConfig)
        for param in self.bert.parameters():
            param.requires_grad = True
        bert_outputdim = self.bert.config.hidden_size
        self.fc = nn.Linear(bert_outputdim, 128)

        #------------------------temporal GCN--------------------------
        self.layer_stack_propagation = nn.ModuleList()  # TODO class initiate
        self.layer_stack_knowledge = nn.ModuleList()
        self.bn_stack_propagation = nn.ModuleList()
        self.bn_stack_knowledge = nn.ModuleList()
        self.temporal_cells = nn.ModuleList()
        for i in range(n_hidden):
            self.layer_stack_propagation.append(PropagationGraphConvLayer_nBatch_geometric(self.n_feature, self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_propagation.append(nn.BatchNorm1d(self.n_feature))
            self.layer_stack_knowledge.append(KnowledgeGraphConvLayer_nBatch_geometric_1(self.n_feature,self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_knowledge.append(nn.BatchNorm1d(self.n_feature))
            self.temporal_cells.append(TemporalEncoding_nBatch_geometric_propagation(self.n_feature, self.n_feature))


        self.mean = Mean_nBatch_geometric(self.device,self.n_feature,self.n_feature)
        # self.linear = nn.Linear(self.n_feature,self.n_output)
        self.fc_rumor_1 = nn.Linear(256,100)
        self.fc_rumor_2 = nn.Linear(100,2)

    def forward(self, data):
        idx2encoding = {}
        features_propagation = []
        features_knowledge = []
        # text_encoding_bert = []

        #将图节点tensor转换为list
        idx_propagation_list = data.x.squeeze(1).cpu().numpy().tolist()
        idx_knowledge_list = data.x_knowledge_idx.squeeze(1).cpu().numpy().tolist()
        # print('---model idx_propagation_list---:', idx_propagation_list)

        #将root_idx对应node2idx转换为在图中的
        root_idx_p = []
        root_idx_k = []
        for i,root_idx in enumerate(data.root_idx):
            root_idx_item = root_idx.item()
            # bert
            input_ids = self.mid2bert_tokenizer[root_idx_item]['input_ids'].to(self.device)
            attention_mask_bert = self.mid2bert_tokenizer[root_idx_item]['attention_mask'].to(self.device)
            text_encoding = self.bert(input_ids,attention_mask=attention_mask_bert)
            text_encoding = text_encoding[0]
            text_encoding = torch.mean(text_encoding, dim=1,keepdim=True)
            text_encoding = text_encoding.squeeze(1)
            text_encoding = self.fc(text_encoding)
            # print('models encoding:',text_encoding) #1,1,768
            if i == 0:
                text_encoding_bert = text_encoding
            else:
                text_encoding_bert = torch.cat([text_encoding_bert,text_encoding],dim=0)

            root_idx_p.append(idx_propagation_list.index(root_idx_item))
            root_idx_k.append(idx_knowledge_list.index(root_idx_item))
        root_idx_p = torch.tensor(root_idx_p)
        root_idx_k = torch.tensor(root_idx_k)
        # print('---model root_idx_p---:',root_idx_p)


        for idx in data.x:
            idx = idx.item()
            text_embedding = self.embedding(torch.LongTensor([idx]).to(self.device))
            # text_embedding = torch.unsqueeze(text_embedding,dim=1)
            features_propagation.append(text_embedding.detach().cpu().numpy())


        # text_encoding_all = torch.tensor(text_encoding_bert)
        # print('models encoding_all:', text_encoding_bert.size())#8,128

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>comment>>>>>>>>>>>>>>>>>>>
        # for idx in
        dual_graph_mutual_index_p = []
        dual_graph_mutual_index_k = []
        for idx in data.x_knowledge_idx:
            idx = idx.item()
            token_idx = torch.LongTensor([idx]).to(self.device)
            entity_embedding = self.embedding(token_idx)
            features_knowledge.append(entity_embedding.detach().cpu().numpy())
            if idx in self.mid2bert_tokenizer:
                dual_graph_mutual_index_p.append(idx_propagation_list.index(idx))
                dual_graph_mutual_index_k.append(idx_knowledge_list.index(idx))
                # features_knowledge.append(idx2encoding[idx].detach().cpu().numpy())
        # print('model dual_graph_mutual_index_p:',dual_graph_mutual_index_p)
        # print('model dual_graph_mutual_index_k:',dual_graph_mutual_index_k)
        features_propagation = torch.tensor(features_propagation)
        features_knowledge = torch.tensor(features_knowledge)

        # emb_propagation = self.embedding(vertices_propagation)
        if self.instance_norm:
            features_propagation = self.norm(features_propagation)

        # emb_knowledge = self.embedding(vertices_knowledge)
        if self.instance_norm:
            features_knowledge = self.norm(features_knowledge)

        x_propagation = features_propagation
        x_knowledge = features_knowledge
        # print('x_propagation:',features_propagation.size()) #7,1,128
        # print('x_knowleddge:',features_knowledge.size()) #771,1,128
        x_propagation = torch.squeeze(x_propagation,dim=1)
        x_knowledge = torch.squeeze(x_knowledge,dim=1)
        # print('x_propagation:', x_propagation.size())  # 7,1,128
        # print('x_knowleddge:', x_knowledge.size())  # 771,1,128
        # print('adj_propagation:',adjs_propagation[0].size())
        # print('adj_knowledge:',adjs_knowledge[0].size())

        x_propagation = x_propagation.to(self.device)
        x_knowledge = x_knowledge.to(self.device)
        x_propagation_node_indices = [data.edge_index,data.x_propagation_node_indices_1_edge_index,data.x_propagation_node_indices_2_edge_index]
        x_propagation_node_values = [data.edge_values,data.x_propagation_node_values_1,data.x_propagation_node_values_2]
        x_progation_edge_num = [data.edge_num,data.x_propagation_edge_num_1,data.x_propagation_edge_num_2]

        x_knowledge_node_indices = [data.x_knowledge_node_indices_0_edge_index, data.x_knowledge_node_indices_1_edge_index,
                                      data.x_knowledge_node_indices_2_edge_index]
        x_knowledge_node_values = [data.x_knowledge_node_values_0, data.x_knowledge_node_values_1,
                                     data.x_knowledge_node_values_2]
        x_knowledge_edge_num = [data.x_knowledge_edge_num_0, data.x_knowledge_edge_num_1,
                                data.x_knowledge_edge_num_2]
        # r = []
        # TODO: dynamic_graph那一篇paper 代码和文中所提不太一样
        last_x = x_propagation
        for i, gcn_layer in enumerate(self.layer_stack_propagation):
            # last_x = x_propagation
            x_gcn = gcn_layer(x_propagation, x_propagation_node_indices[i], x_propagation_node_values[i], root_idx_p, data.x_propagation_node_num, x_progation_edge_num[i],data.batch)
            # print('x_gcn:',x_gcn.size())
            x_propagation = self.bn_stack_propagation[i](x_gcn)
            # x_knowledge = self.bn_stack_knowledge[i](self.layer_stack_knowledge[i](x_knowledge,x_knowledge_node_indices[i],x_knowledge_node_values[i], root_idx_k,data.x_knowledge_node_num,x_knowledge_edge_num[i],data.batch))
            # if i < len(self.layer_stack) - 1:
            #====================================
            x_propagation = F.leaky_relu(x_propagation)
            # x_knowledge = F.leaky_relu(x_knowledge)

            x_propagation = F.dropout(x_propagation, self.dropout, training=self.training)
            # x_knowledge = F.dropout(x_knowledge,self.dropout,training = self.training)
            #====================================
            # print('attention test models:')
            # print('x_propagation size:',x_propagation.size())
            # print('x_knowledge size:',x_knowledge.size())
            x_propagation = self.temporal_cells[i](x_propagation, x_knowledge,dual_graph_mutual_index_p,dual_graph_mutual_index_k,last_x,data.x_propagation_node_num,data.x_knowledge_node_num)  # temporal encoding
        #=========================================
        # x_propagation = F.relu(x_propagation)
        #===========================================
        # x_knowledge = F.relu(x_knowledge)
        # x = torch.cat([x_propagation,x_knowledge],1)
        # print('---model x propagation---:', x_propagation.size())
        x = self.mean(x_propagation, data.batch)
        x = x.squeeze(1)
        # print('model x size:',x.size())
        #TODO: x,text_encoding normalized(?)
        x_fusion = torch.cat([x,text_encoding_bert],dim=1)
        # print('---model x ---:', x.size()) #[32,128]
        x_fusion = F.leaky_relu(self.fc_rumor_1(x_fusion))
        x_fusion = self.fc_rumor_2(x_fusion)
        # x = self.mask(x, vertices)
        # print('----result before sigmoid----:',x)
        x_fusion = torch.sigmoid(x_fusion)
        # print('----result after sigmoid----:',x,x.size())
        return x_fusion
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class dual_DynamicGCN_nBatch_knowledge(nn.Module):
    def __init__(self, token_dict, mid2bert_tokenizer, bert_path, n_output,config,device, n_hidden=3, dropout=0.2, instance_norm=False):
        super(dual_DynamicGCN_nBatch_knowledge, self).__init__()
        self.token_dict = token_dict
        self.mid2bert_tokenizer = mid2bert_tokenizer
        self.config = config
        self.dropout = dropout
        self.instance_norm = instance_norm
        if self.instance_norm:
            self.norm = nn.InstanceNorm1d(self.config.graph_embedding_dim, momentum=0.0, affine=True)
        self.n_output = n_output
        self.device = device
        self.text_length = torch.tensor([self.config.text_max_length]).to(self.device)

        #-----------------------text-----------------------------------
        self.embedding = nn.Embedding(len(self.token_dict), config.graph_embedding_dim)
        self.gru = GRUEncoder(self.config.graph_embedding_dim, self.config.hidden_dim)
        # self.embedding.weight = nn.Parameter(pretrained_emb)
        # self.embedding.weight.requires_grad = False

        self.n_feature = self.config.hidden_dim
        # self.n_input = pretrained_emb.size(0)

        #------------------ ----bert-----------------------------------
        modelConfig = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path, config=modelConfig)
        for param in self.bert.parameters():
            param.requires_grad = True
        bert_outputdim = self.bert.config.hidden_size
        self.fc = nn.Linear(bert_outputdim, 128)

        #------------------------temporal GCN--------------------------
        self.layer_stack_propagation = nn.ModuleList()  # TODO class initiate
        self.layer_stack_knowledge = nn.ModuleList()
        self.bn_stack_propagation = nn.ModuleList()
        self.bn_stack_knowledge = nn.ModuleList()
        self.temporal_cells = nn.ModuleList()
        for i in range(n_hidden):
            self.layer_stack_propagation.append(PropagationGraphConvLayer_nBatch_geometric(self.n_feature, self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_propagation.append(nn.BatchNorm1d(self.n_feature))
            self.layer_stack_knowledge.append(KnowledgeGraphConvLayer_nBatch_geometric_1(self.n_feature,self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_knowledge.append(nn.BatchNorm1d(self.n_feature))
            self.temporal_cells.append(TemporalEncoding_nBatch_geometric_knowledge(self.n_feature, self.n_feature))


        self.mean = Mean_nBatch_geometric(self.device,self.n_feature,self.n_feature)
        # self.linear = nn.Linear(self.n_feature,self.n_output)
        self.fc_rumor_1 = nn.Linear(256,100)
        self.fc_rumor_2 = nn.Linear(100,2)

    def forward(self, data):
        idx2encoding = {}
        features_propagation = []
        features_knowledge = []
        # text_encoding_bert = []

        #将图节点tensor转换为list
        idx_propagation_list = data.x.squeeze(1).cpu().numpy().tolist()
        idx_knowledge_list = data.x_knowledge_idx.squeeze(1).cpu().numpy().tolist()
        # print('---model idx_propagation_list---:', idx_propagation_list)

        #将root_idx对应node2idx转换为在图中的
        root_idx_p = []
        root_idx_k = []
        for i,root_idx in enumerate(data.root_idx):
            root_idx_item = root_idx.item()
            # bert
            input_ids = self.mid2bert_tokenizer[root_idx_item]['input_ids'].to(self.device)
            attention_mask_bert = self.mid2bert_tokenizer[root_idx_item]['attention_mask'].to(self.device)
            text_encoding = self.bert(input_ids,attention_mask=attention_mask_bert)
            text_encoding = text_encoding[0]
            text_encoding = torch.mean(text_encoding, dim=1,keepdim=True)
            text_encoding = text_encoding.squeeze(1)
            text_encoding = self.fc(text_encoding)
            # print('models encoding:',text_encoding) #1,1,768
            if i == 0:
                text_encoding_bert = text_encoding
            else:
                text_encoding_bert = torch.cat([text_encoding_bert,text_encoding],dim=0)

            root_idx_p.append(idx_propagation_list.index(root_idx_item))
            root_idx_k.append(idx_knowledge_list.index(root_idx_item))
        root_idx_p = torch.tensor(root_idx_p)
        root_idx_k = torch.tensor(root_idx_k)
        # print('---model root_idx_p---:',root_idx_p)


        for idx in data.x:
            idx = idx.item()
            text_embedding = self.embedding(torch.LongTensor([idx]).to(self.device))
            # text_embedding = torch.unsqueeze(text_embedding,dim=1)
            features_propagation.append(text_embedding.detach().cpu().numpy())


        # text_encoding_all = torch.tensor(text_encoding_bert)
        # print('models encoding_all:', text_encoding_bert.size())#8,128

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>comment>>>>>>>>>>>>>>>>>>>
        # for idx in
        dual_graph_mutual_index_p = []
        dual_graph_mutual_index_k = []
        for idx in data.x_knowledge_idx:
            idx = idx.item()
            token_idx = torch.LongTensor([idx]).to(self.device)
            entity_embedding = self.embedding(token_idx)
            features_knowledge.append(entity_embedding.detach().cpu().numpy())
            if idx in self.mid2bert_tokenizer:
                dual_graph_mutual_index_p.append(idx_propagation_list.index(idx))
                dual_graph_mutual_index_k.append(idx_knowledge_list.index(idx))
                # features_knowledge.append(idx2encoding[idx].detach().cpu().numpy())
        # print('model dual_graph_mutual_index_p:',dual_graph_mutual_index_p)
        # print('model dual_graph_mutual_index_k:',dual_graph_mutual_index_k)
        features_propagation = torch.tensor(features_propagation)
        features_knowledge = torch.tensor(features_knowledge)

        # emb_propagation = self.embedding(vertices_propagation)
        if self.instance_norm:
            features_propagation = self.norm(features_propagation)

        # emb_knowledge = self.embedding(vertices_knowledge)
        if self.instance_norm:
            features_knowledge = self.norm(features_knowledge)

        x_propagation = features_propagation
        x_knowledge = features_knowledge
        # print('x_propagation:',features_propagation.size()) #7,1,128
        # print('x_knowleddge:',features_knowledge.size()) #771,1,128
        x_propagation = torch.squeeze(x_propagation,dim=1)
        x_knowledge = torch.squeeze(x_knowledge,dim=1)
        # print('x_propagation:', x_propagation.size())  # 7,1,128
        # print('x_knowleddge:', x_knowledge.size())  # 771,1,128
        # print('adj_propagation:',adjs_propagation[0].size())
        # print('adj_knowledge:',adjs_knowledge[0].size())

        x_propagation = x_propagation.to(self.device)
        x_knowledge = x_knowledge.to(self.device)
        x_propagation_node_indices = [data.edge_index,data.x_propagation_node_indices_1_edge_index,data.x_propagation_node_indices_2_edge_index]
        x_propagation_node_values = [data.edge_values,data.x_propagation_node_values_1,data.x_propagation_node_values_2]
        x_progation_edge_num = [data.edge_num,data.x_propagation_edge_num_1,data.x_propagation_edge_num_2]

        x_knowledge_node_indices = [data.x_knowledge_node_indices_0_edge_index, data.x_knowledge_node_indices_1_edge_index,
                                      data.x_knowledge_node_indices_2_edge_index]
        x_knowledge_node_values = [data.x_knowledge_node_values_0, data.x_knowledge_node_values_1,
                                     data.x_knowledge_node_values_2]
        x_knowledge_edge_num = [data.x_knowledge_edge_num_0, data.x_knowledge_edge_num_1,
                                data.x_knowledge_edge_num_2]
        # r = []
        # TODO: dynamic_graph那一篇paper 代码和文中所提不太一样
        last_x = x_knowledge
        for i, gcn_layer in enumerate(self.layer_stack_propagation):
            # last_x = x_propagation
            # x_gcn = gcn_layer(x_propagation, x_propagation_node_indices[i], x_propagation_node_values[i], root_idx_p, data.x_propagation_node_num, x_progation_edge_num[i],data.batch)
            # print('x_gcn:',x_gcn.size())
            # x_propagation = self.bn_stack_propagation[i](x_gcn)
            x_knowledge = self.bn_stack_knowledge[i](self.layer_stack_knowledge[i](x_knowledge,x_knowledge_node_indices[i],x_knowledge_node_values[i], root_idx_k,data.x_knowledge_node_num,x_knowledge_edge_num[i],data.batch))
            # if i < len(self.layer_stack) - 1:
            #====================================
            # x_propagation = F.leaky_relu(x_propagation)
            x_knowledge = F.leaky_relu(x_knowledge)

            # x_propagation = F.dropout(x_propagation, self.dropout, training=self.training)
            x_knowledge = F.dropout(x_knowledge,self.dropout,training = self.training)
            #====================================
            # print('attention test models:')
            # print('x_propagation size:',x_propagation.size())
            # print('x_knowledge size:',x_knowledge.size())
            x_propagation, x_knowledge = self.temporal_cells[i](x_propagation, x_knowledge,dual_graph_mutual_index_p,dual_graph_mutual_index_k,last_x,data.x_propagation_node_num,data.x_knowledge_node_num)  # temporal encoding
        #=========================================
        # x_propagation = F.relu(x_propagation)
        #===========================================
        # x_knowledge = F.relu(x_knowledge)
        # x = torch.cat([x_propagation,x_knowledge],1)
        # print('---model x propagation---:', x_propagation.size())
        x = self.mean(x_propagation, data.batch)
        x = x.squeeze(1)
        # print('model x size:',x.size())
        #TODO: x,text_encoding normalized(?)
        x_fusion = torch.cat([x,text_encoding_bert],dim=1)
        # print('---model x ---:', x.size()) #[32,128]
        x_fusion = F.leaky_relu(self.fc_rumor_1(x_fusion))
        x_fusion = self.fc_rumor_2(x_fusion)
        # x = self.mask(x, vertices)
        # print('----result before sigmoid----:',x)
        x_fusion = torch.sigmoid(x_fusion)
        # print('----result after sigmoid----:',x,x.size())
        return x_fusion
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class dual_DynamicGCN_nBatch_completed_wotf(nn.Module):
    def __init__(self, token_dict, mid2bert_tokenizer, bert_path, n_output,config,device, n_hidden=3, dropout=0.2, instance_norm=False):
        super(dual_DynamicGCN_nBatch_completed_wotf, self).__init__()
        self.token_dict = token_dict
        self.mid2bert_tokenizer = mid2bert_tokenizer
        self.config = config
        self.dropout = dropout
        self.instance_norm = instance_norm
        if self.instance_norm:
            self.norm = nn.InstanceNorm1d(self.config.graph_embedding_dim, momentum=0.0, affine=True)
        self.n_output = n_output
        self.device = device
        self.text_length = torch.tensor([self.config.text_max_length]).to(self.device)

        #-----------------------text-----------------------------------
        self.embedding = nn.Embedding(len(self.token_dict), config.graph_embedding_dim)
        self.gru = GRUEncoder(self.config.graph_embedding_dim, self.config.hidden_dim)
        # self.embedding.weight = nn.Parameter(pretrained_emb)
        # self.embedding.weight.requires_grad = False

        self.n_feature = self.config.hidden_dim
        # self.n_input = pretrained_emb.size(0)

        #------------------ ----bert-----------------------------------
        modelConfig = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path, config=modelConfig)
        for param in self.bert.parameters():
            param.requires_grad = True
        bert_outputdim = self.bert.config.hidden_size
        self.fc = nn.Linear(bert_outputdim, 128)

        #------------------------temporal GCN--------------------------
        self.layer_stack_propagation = nn.ModuleList()  # TODO class initiate
        self.layer_stack_knowledge = nn.ModuleList()
        self.bn_stack_propagation = nn.ModuleList()
        self.bn_stack_knowledge = nn.ModuleList()
        self.temporal_cells = nn.ModuleList()
        for i in range(n_hidden):
            self.layer_stack_propagation.append(PropagationGraphConvLayer_nBatch_geometric(self.n_feature, self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_propagation.append(nn.BatchNorm1d(self.n_feature))
            self.layer_stack_knowledge.append(KnowledgeGraphConvLayer_nBatch_geometric_1(self.n_feature,self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_knowledge.append(nn.BatchNorm1d(self.n_feature))
            self.temporal_cells.append(TemporalEncoding_nBatch_geometric_wotf(self.n_feature, self.n_feature))


        self.mean = Mean_nBatch_geometric(self.device,self.n_feature,self.n_feature)
        # self.linear = nn.Linear(self.n_feature,self.n_output)
        self.fc_rumor_1 = nn.Linear(256,100)
        self.fc_rumor_2 = nn.Linear(100,2)

    def forward(self, data):
        idx2encoding = {}
        features_propagation = []
        features_knowledge = []
        # text_encoding_bert = []

        #将图节点tensor转换为list
        idx_propagation_list = data.x.squeeze(1).cpu().numpy().tolist()
        idx_knowledge_list = data.x_knowledge_idx.squeeze(1).cpu().numpy().tolist()
        # print('---model idx_propagation_list---:', idx_propagation_list)

        #将root_idx对应node2idx转换为在图中的
        root_idx_p = []
        root_idx_k = []
        for i,root_idx in enumerate(data.root_idx):
            root_idx_item = root_idx.item()
            # bert
            input_ids = self.mid2bert_tokenizer[root_idx_item]['input_ids'].to(self.device)
            attention_mask_bert = self.mid2bert_tokenizer[root_idx_item]['attention_mask'].to(self.device)
            text_encoding = self.bert(input_ids,attention_mask=attention_mask_bert)
            text_encoding = text_encoding[0]
            text_encoding = torch.mean(text_encoding, dim=1,keepdim=True)
            text_encoding = text_encoding.squeeze(1)
            text_encoding = self.fc(text_encoding)
            # print('models encoding:',text_encoding) #1,1,768
            if i == 0:
                text_encoding_bert = text_encoding
            else:
                text_encoding_bert = torch.cat([text_encoding_bert,text_encoding],dim=0)

            root_idx_p.append(idx_propagation_list.index(root_idx_item))
            root_idx_k.append(idx_knowledge_list.index(root_idx_item))
        root_idx_p = torch.tensor(root_idx_p)
        root_idx_k = torch.tensor(root_idx_k)
        # print('---model root_idx_p---:',root_idx_p)


        for idx in data.x:
            idx = idx.item()
            text_embedding = self.embedding(torch.LongTensor([idx]).to(self.device))
            # text_embedding = torch.unsqueeze(text_embedding,dim=1)
            features_propagation.append(text_embedding.detach().cpu().numpy())


        # text_encoding_all = torch.tensor(text_encoding_bert)
        # print('models encoding_all:', text_encoding_bert.size())#8,128

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>comment>>>>>>>>>>>>>>>>>>>
        # for idx in
        dual_graph_mutual_index_p = []
        dual_graph_mutual_index_k = []
        for idx in data.x_knowledge_idx:
            idx = idx.item()
            token_idx = torch.LongTensor([idx]).to(self.device)
            entity_embedding = self.embedding(token_idx)
            features_knowledge.append(entity_embedding.detach().cpu().numpy())
            if idx in self.mid2bert_tokenizer:
                dual_graph_mutual_index_p.append(idx_propagation_list.index(idx))
                dual_graph_mutual_index_k.append(idx_knowledge_list.index(idx))
                # features_knowledge.append(idx2encoding[idx].detach().cpu().numpy())
        # print('model dual_graph_mutual_index_p:',dual_graph_mutual_index_p)
        # print('model dual_graph_mutual_index_k:',dual_graph_mutual_index_k)
        features_propagation = torch.tensor(features_propagation)
        features_knowledge = torch.tensor(features_knowledge)

        # emb_propagation = self.embedding(vertices_propagation)
        if self.instance_norm:
            features_propagation = self.norm(features_propagation)

        # emb_knowledge = self.embedding(vertices_knowledge)
        if self.instance_norm:
            features_knowledge = self.norm(features_knowledge)

        x_propagation = features_propagation
        x_knowledge = features_knowledge
        # print('x_propagation:',features_propagation.size()) #7,1,128
        # print('x_knowleddge:',features_knowledge.size()) #771,1,128
        x_propagation = torch.squeeze(x_propagation,dim=1)
        x_knowledge = torch.squeeze(x_knowledge,dim=1)
        # print('x_propagation:', x_propagation.size())  # 7,1,128
        # print('x_knowleddge:', x_knowledge.size())  # 771,1,128
        # print('adj_propagation:',adjs_propagation[0].size())
        # print('adj_knowledge:',adjs_knowledge[0].size())

        x_propagation = x_propagation.to(self.device)
        x_knowledge = x_knowledge.to(self.device)
        x_propagation_node_indices = [data.edge_index,data.x_propagation_node_indices_1_edge_index,data.x_propagation_node_indices_2_edge_index]
        x_propagation_node_values = [data.edge_values,data.x_propagation_node_values_1,data.x_propagation_node_values_2]
        x_progation_edge_num = [data.edge_num,data.x_propagation_edge_num_1,data.x_propagation_edge_num_2]

        x_knowledge_node_indices = [data.x_knowledge_node_indices_0_edge_index, data.x_knowledge_node_indices_1_edge_index,
                                      data.x_knowledge_node_indices_2_edge_index]
        x_knowledge_node_values = [data.x_knowledge_node_values_0, data.x_knowledge_node_values_1,
                                     data.x_knowledge_node_values_2]
        x_knowledge_edge_num = [data.x_knowledge_edge_num_0, data.x_knowledge_edge_num_1,
                                data.x_knowledge_edge_num_2]
        # r = []
        # TODO: dynamic_graph那一篇paper 代码和文中所提不太一样
        last_x = x_knowledge
        for i, gcn_layer in enumerate(self.layer_stack_propagation):
            # last_x = x_propagation
            x_gcn = gcn_layer(x_propagation, x_propagation_node_indices[i], x_propagation_node_values[i], root_idx_p, data.x_propagation_node_num, x_progation_edge_num[i],data.batch)
            # print('x_gcn:',x_gcn.size())
            x_propagation = self.bn_stack_propagation[i](x_gcn)
            x_knowledge = self.bn_stack_knowledge[i](self.layer_stack_knowledge[i](x_knowledge,x_knowledge_node_indices[i],x_knowledge_node_values[i], root_idx_k,data.x_knowledge_node_num,x_knowledge_edge_num[i],data.batch))
            # if i < len(self.layer_stack) - 1:
            #====================================
            x_propagation = F.leaky_relu(x_propagation)
            x_knowledge = F.leaky_relu(x_knowledge)

            x_propagation = F.dropout(x_propagation, self.dropout, training=self.training)
            x_knowledge = F.dropout(x_knowledge,self.dropout,training = self.training)
            #====================================
            # print('attention test models:')
            # print('x_propagation size:',x_propagation.size())
            # print('x_knowledge size:',x_knowledge.size())
            # x_propagation, x_knowledge = self.temporal_cells[i](x_propagation, x_knowledge,dual_graph_mutual_index_p,dual_graph_mutual_index_k,last_x,data.x_propagation_node_num,data.x_knowledge_node_num)  # temporal encoding
        #=========================================
        # x_propagation = F.relu(x_propagation)
        #===========================================
        # x_knowledge = F.relu(x_knowledge)
        # x = torch.cat([x_propagation,x_knowledge],1)
        # print('---model x propagation---:', x_propagation.size())
        x_propagation, x_knowledge = self.temporal_cells[0](x_propagation, x_knowledge, dual_graph_mutual_index_p,
                                                            dual_graph_mutual_index_k, last_x,
                                                            data.x_propagation_node_num,
                                                            data.x_knowledge_node_num)  # temporal encoding
        x = self.mean(x_propagation, data.batch)
        x = x.squeeze(1)
        # print('model x size:',x.size())
        #TODO: x,text_encoding normalized(?)
        x_fusion = torch.cat([x,text_encoding_bert],dim=1)
        # print('---model x ---:', x.size()) #[32,128]
        x_fusion = F.leaky_relu(self.fc_rumor_1(x_fusion))
        x_fusion = self.fc_rumor_2(x_fusion)
        # x = self.mask(x, vertices)
        # print('----result before sigmoid----:',x)
        x_fusion = torch.sigmoid(x_fusion)
        # print('----result after sigmoid----:',x,x.size())
        return x_fusion
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class dual_DynamicGCN_nBatch_completed_static(nn.Module):
    def __init__(self, token_dict, mid2bert_tokenizer, bert_path, n_output,config,device, n_hidden=3, dropout=0.2, instance_norm=False):
        super(dual_DynamicGCN_nBatch_completed_static, self).__init__()
        self.token_dict = token_dict
        self.mid2bert_tokenizer = mid2bert_tokenizer
        self.config = config
        self.dropout = dropout
        self.instance_norm = instance_norm
        if self.instance_norm:
            self.norm = nn.InstanceNorm1d(self.config.graph_embedding_dim, momentum=0.0, affine=True)
        self.n_output = n_output
        self.device = device
        self.text_length = torch.tensor([self.config.text_max_length]).to(self.device)

        #-----------------------text-----------------------------------
        self.embedding = nn.Embedding(len(self.token_dict), config.graph_embedding_dim)
        self.gru = GRUEncoder(self.config.graph_embedding_dim, self.config.hidden_dim)
        # self.embedding.weight = nn.Parameter(pretrained_emb)
        # self.embedding.weight.requires_grad = False

        self.n_feature = self.config.hidden_dim
        # self.n_input = pretrained_emb.size(0)

        #------------------ ----bert-----------------------------------
        modelConfig = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path, config=modelConfig)
        for param in self.bert.parameters():
            param.requires_grad = True
        bert_outputdim = self.bert.config.hidden_size
        self.fc = nn.Linear(bert_outputdim, 128)

        #------------------------temporal GCN--------------------------
        self.layer_stack_propagation = nn.ModuleList()  # TODO class initiate
        self.layer_stack_knowledge = nn.ModuleList()
        self.bn_stack_propagation = nn.ModuleList()
        self.bn_stack_knowledge = nn.ModuleList()
        self.temporal_cells = nn.ModuleList()
        for i in range(n_hidden):
            self.layer_stack_propagation.append(PropagationGraphConvLayer_nBatch_geometric(self.n_feature, self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_propagation.append(nn.BatchNorm1d(self.n_feature))
            self.layer_stack_knowledge.append(KnowledgeGraphConvLayer_nBatch_geometric_1(self.n_feature,self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_knowledge.append(nn.BatchNorm1d(self.n_feature))
            self.temporal_cells.append(TemporalEncoding_nBatch_geometric_1(self.n_feature, self.n_feature))


        self.mean = Mean_nBatch_geometric(self.device,self.n_feature,self.n_feature)
        # self.linear = nn.Linear(self.n_feature,self.n_output)
        self.fc_rumor_1 = nn.Linear(256,100)
        self.fc_rumor_2 = nn.Linear(100,2)

    def forward(self, data):
        idx2encoding = {}
        features_propagation = []
        features_knowledge = []
        # text_encoding_bert = []

        #将图节点tensor转换为list
        idx_propagation_list = data.x.squeeze(1).cpu().numpy().tolist()
        idx_knowledge_list = data.x_knowledge_idx.squeeze(1).cpu().numpy().tolist()
        # print('---model idx_propagation_list---:', idx_propagation_list)

        #将root_idx对应node2idx转换为在图中的
        root_idx_p = []
        root_idx_k = []
        for i,root_idx in enumerate(data.root_idx):
            root_idx_item = root_idx.item()
            # bert
            input_ids = self.mid2bert_tokenizer[root_idx_item]['input_ids'].to(self.device)
            attention_mask_bert = self.mid2bert_tokenizer[root_idx_item]['attention_mask'].to(self.device)
            text_encoding = self.bert(input_ids,attention_mask=attention_mask_bert)
            text_encoding = text_encoding[0]
            text_encoding = torch.mean(text_encoding, dim=1,keepdim=True)
            text_encoding = text_encoding.squeeze(1)
            text_encoding = self.fc(text_encoding)
            # print('models encoding:',text_encoding) #1,1,768
            if i == 0:
                text_encoding_bert = text_encoding
            else:
                text_encoding_bert = torch.cat([text_encoding_bert,text_encoding],dim=0)

            root_idx_p.append(idx_propagation_list.index(root_idx_item))
            root_idx_k.append(idx_knowledge_list.index(root_idx_item))
        root_idx_p = torch.tensor(root_idx_p)
        root_idx_k = torch.tensor(root_idx_k)
        # print('---model root_idx_p---:',root_idx_p)


        for idx in data.x:
            idx = idx.item()
            text_embedding = self.embedding(torch.LongTensor([idx]).to(self.device))
            # text_embedding = torch.unsqueeze(text_embedding,dim=1)
            features_propagation.append(text_embedding.detach().cpu().numpy())


        # text_encoding_all = torch.tensor(text_encoding_bert)
        # print('models encoding_all:', text_encoding_bert.size())#8,128

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>comment>>>>>>>>>>>>>>>>>>>
        # for idx in
        dual_graph_mutual_index_p = []
        dual_graph_mutual_index_k = []
        for idx in data.x_knowledge_idx:
            idx = idx.item()
            token_idx = torch.LongTensor([idx]).to(self.device)
            entity_embedding = self.embedding(token_idx)
            features_knowledge.append(entity_embedding.detach().cpu().numpy())
            if idx in self.mid2bert_tokenizer:
                dual_graph_mutual_index_p.append(idx_propagation_list.index(idx))
                dual_graph_mutual_index_k.append(idx_knowledge_list.index(idx))
                # features_knowledge.append(idx2encoding[idx].detach().cpu().numpy())
        # print('model dual_graph_mutual_index_p:',dual_graph_mutual_index_p)
        # print('model dual_graph_mutual_index_k:',dual_graph_mutual_index_k)
        features_propagation = torch.tensor(features_propagation)
        features_knowledge = torch.tensor(features_knowledge)

        # emb_propagation = self.embedding(vertices_propagation)
        if self.instance_norm:
            features_propagation = self.norm(features_propagation)

        # emb_knowledge = self.embedding(vertices_knowledge)
        if self.instance_norm:
            features_knowledge = self.norm(features_knowledge)

        x_propagation = features_propagation
        x_knowledge = features_knowledge
        # print('x_propagation:',features_propagation.size()) #7,1,128
        # print('x_knowleddge:',features_knowledge.size()) #771,1,128
        x_propagation = torch.squeeze(x_propagation,dim=1)
        x_knowledge = torch.squeeze(x_knowledge,dim=1)
        # print('x_propagation:', x_propagation.size())  # 7,1,128
        # print('x_knowleddge:', x_knowledge.size())  # 771,1,128
        # print('adj_propagation:',adjs_propagation[0].size())
        # print('adj_knowledge:',adjs_knowledge[0].size())

        x_propagation = x_propagation.to(self.device)
        x_knowledge = x_knowledge.to(self.device)
        x_propagation_node_indices = [data.edge_index,data.x_propagation_node_indices_1_edge_index,data.x_propagation_node_indices_2_edge_index]
        x_propagation_node_values = [data.edge_values,data.x_propagation_node_values_1,data.x_propagation_node_values_2]
        x_progation_edge_num = [data.edge_num,data.x_propagation_edge_num_1,data.x_propagation_edge_num_2]

        x_knowledge_node_indices = [data.x_knowledge_node_indices_0_edge_index, data.x_knowledge_node_indices_1_edge_index,
                                      data.x_knowledge_node_indices_2_edge_index]
        x_knowledge_node_values = [data.x_knowledge_node_values_0, data.x_knowledge_node_values_1,
                                     data.x_knowledge_node_values_2]
        x_knowledge_edge_num = [data.x_knowledge_edge_num_0, data.x_knowledge_edge_num_1,
                                data.x_knowledge_edge_num_2]
        # r = []
        # TODO: dynamic_graph那一篇paper 代码和文中所提不太一样
        last_x = x_knowledge
        # for i, gcn_layer in enumerate(self.layer_stack_propagation):
        # last_x = x_propagation
        x_gcn = self.layer_stack_propagation[2](x_propagation, x_propagation_node_indices[2], x_propagation_node_values[2], root_idx_p, data.x_propagation_node_num, x_progation_edge_num[2],data.batch)
        # print('x_gcn:',x_gcn.size())
        x_propagation = self.bn_stack_propagation[2](x_gcn)
        x_knowledge = self.bn_stack_knowledge[2](self.layer_stack_knowledge[2](x_knowledge,x_knowledge_node_indices[2],x_knowledge_node_values[2], root_idx_k,data.x_knowledge_node_num,x_knowledge_edge_num[2],data.batch))
        # if i < len(self.layer_stack) - 1:
        #====================================
        x_propagation = F.leaky_relu(x_propagation)
        x_knowledge = F.leaky_relu(x_knowledge)

        x_propagation = F.dropout(x_propagation, self.dropout, training=self.training)
        x_knowledge = F.dropout(x_knowledge,self.dropout,training = self.training)
        #====================================
        # print('attention test models:')
        # print('x_propagation size:',x_propagation.size())
        # print('x_knowledge size:',x_knowledge.size())
        x_propagation, x_knowledge = self.temporal_cells[2](x_propagation, x_knowledge,dual_graph_mutual_index_p,dual_graph_mutual_index_k,last_x,data.x_propagation_node_num,data.x_knowledge_node_num)  # temporal encoding
        #=========================================
        # x_propagation = F.relu(x_propagation)
        #===========================================
        # x_knowledge = F.relu(x_knowledge)
        # x = torch.cat([x_propagation,x_knowledge],1)
        # print('---model x propagation---:', x_propagation.size())
        x = self.mean(x_propagation, data.batch)
        x = x.squeeze(1)
        # print('model x size:',x.size())
        #TODO: x,text_encoding normalized(?)
        x_fusion = torch.cat([x,text_encoding_bert],dim=1)
        # print('---model x ---:', x.size()) #[32,128]
        x_fusion = F.leaky_relu(self.fc_rumor_1(x_fusion))
        x_fusion = self.fc_rumor_2(x_fusion)
        # x = self.mask(x, vertices)
        # print('----result before sigmoid----:',x)
        x_fusion = torch.sigmoid(x_fusion)
        # print('----result after sigmoid----:',x,x.size())
        return x_fusion
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


#只对propagation图进行了root_enhance||fusion模块中knowledge图只对post节点拼接
class dual_DynamicGCN_nBatch_completed(nn.Module):
    def __init__(self, token_dict, mid2bert_tokenizer, bert_path, n_output,config,device, n_hidden=3, dropout=0.2, instance_norm=False):
        super(dual_DynamicGCN_nBatch_completed, self).__init__()
        self.token_dict = token_dict
        self.mid2bert_tokenizer = mid2bert_tokenizer
        self.config = config
        self.dropout = dropout
        self.instance_norm = instance_norm
        if self.instance_norm:
            self.norm = nn.InstanceNorm1d(self.config.graph_embedding_dim, momentum=0.0, affine=True)
        self.n_output = n_output
        self.device = device
        self.text_length = torch.tensor([self.config.text_max_length]).to(self.device)

        #-----------------------text-----------------------------------
        self.embedding = nn.Embedding(len(self.token_dict), config.graph_embedding_dim)
        self.gru = GRUEncoder(self.config.graph_embedding_dim, self.config.hidden_dim)
        # self.embedding.weight = nn.Parameter(pretrained_emb)
        # self.embedding.weight.requires_grad = False

        self.n_feature = self.config.hidden_dim
        # self.n_input = pretrained_emb.size(0)

        #------------------ ----bert-----------------------------------
        modelConfig = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path, config=modelConfig)
        for param in self.bert.parameters():
            param.requires_grad = True
        bert_outputdim = self.bert.config.hidden_size
        self.fc = nn.Linear(bert_outputdim, 128)


        #------------------------temporal GCN--------------------------
        self.layer_stack_propagation = nn.ModuleList()  # TODO class initiate
        self.layer_stack_knowledge = nn.ModuleList()
        self.bn_stack_propagation = nn.ModuleList()
        self.bn_stack_knowledge = nn.ModuleList()
        self.temporal_cells = nn.ModuleList()
        for i in range(n_hidden):
            self.layer_stack_propagation.append(PropagationGraphConvLayer_nBatch_geometric(self.n_feature, self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_propagation.append(nn.BatchNorm1d(self.n_feature))
            self.layer_stack_knowledge.append(KnowledgeGraphConvLayer_nBatch_geometric(self.n_feature,self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_knowledge.append(nn.BatchNorm1d(self.n_feature))
            self.temporal_cells.append(TemporalEncoding_nBatch_geometric(self.n_feature, self.n_feature))


        self.mean = Mean_nBatch_geometric(self.device,self.n_feature,self.n_feature)
        # self.linear = nn.Linear(self.n_feature,self.n_output)
        self.fc_rumor_1 = nn.Linear(256,100)
        self.fc_rumor_2 = nn.Linear(100,2)

    def forward(self, data):
        idx2encoding = {}
        features_propagation = []
        features_knowledge = []
        # text_encoding_bert = []

        #将图节点tensor转换为list
        idx_propagation_list = data.x.squeeze(1).cpu().numpy().tolist()
        idx_knowledge_list = data.x_knowledge_idx.squeeze(1).cpu().numpy().tolist()
        # print('---model idx_propagation_list---:', idx_propagation_list)

        #将root_idx对应node2idx转换为在图中的
        root_idx_p = []
        for i,root_idx in enumerate(data.root_idx):
            root_idx_item = root_idx.item()
            # bert
            input_ids = self.mid2bert_tokenizer[root_idx_item]['input_ids'].to(self.device)
            attention_mask_bert = self.mid2bert_tokenizer[root_idx_item]['attention_mask'].to(self.device)
            text_encoding = self.bert(input_ids,attention_mask=attention_mask_bert)
            text_encoding = text_encoding[0]
            text_encoding = torch.mean(text_encoding, dim=1,keepdim=True)
            text_encoding = text_encoding.squeeze(1)
            text_encoding = self.fc(text_encoding)
            # print('models encoding:',text_encoding) #1,1,768
            if i == 0:
                text_encoding_bert = text_encoding
            else:
                text_encoding_bert = torch.cat([text_encoding_bert,text_encoding],dim=0)

            root_idx_p.append(idx_propagation_list.index(root_idx_item))
        root_idx_p = torch.tensor(root_idx_p)
        # print('---model root_idx_p---:',root_idx_p)


        for idx in data.x:
            idx = idx.item()
            text_embedding = self.embedding(torch.LongTensor([idx]).to(self.device))
            # text_embedding = torch.unsqueeze(text_embedding,dim=1)
            features_propagation.append(text_embedding.detach().cpu().numpy())


        # text_encoding_all = torch.tensor(text_encoding_bert)
        # print('models encoding_all:', text_encoding_bert.size())#8,128

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>comment>>>>>>>>>>>>>>>>>>>
        # for idx in
        dual_graph_mutual_index_p = []
        dual_graph_mutual_index_k = []
        for idx in data.x_knowledge_idx:
            idx = idx.item()
            token_idx = torch.LongTensor([idx]).to(self.device)
            entity_embedding = self.embedding(token_idx)
            features_knowledge.append(entity_embedding.detach().cpu().numpy())
            if idx in self.mid2bert_tokenizer:
                dual_graph_mutual_index_p.append(idx_propagation_list.index(idx))
                dual_graph_mutual_index_k.append(idx_knowledge_list.index(idx))
                # features_knowledge.append(idx2encoding[idx].detach().cpu().numpy())
        # print('model dual_graph_mutual_index_p:',dual_graph_mutual_index_p)
        # print('model dual_graph_mutual_index_k:',dual_graph_mutual_index_k)
        features_propagation = torch.tensor(features_propagation)
        features_knowledge = torch.tensor(features_knowledge)

        # emb_propagation = self.embedding(vertices_propagation)
        if self.instance_norm:
            features_propagation = self.norm(features_propagation)

        # emb_knowledge = self.embedding(vertices_knowledge)
        if self.instance_norm:
            features_knowledge = self.norm(features_knowledge)

        x_propagation = features_propagation
        x_knowledge = features_knowledge
        # print('x_propagation:',features_propagation.size()) #7,1,128
        # print('x_knowleddge:',features_knowledge.size()) #771,1,128
        x_propagation = torch.squeeze(x_propagation,dim=1)
        x_knowledge = torch.squeeze(x_knowledge,dim=1)
        # print('x_propagation:', x_propagation.size())  # 7,1,128
        # print('x_knowleddge:', x_knowledge.size())  # 771,1,128
        # print('adj_propagation:',adjs_propagation[0].size())
        # print('adj_knowledge:',adjs_knowledge[0].size())

        x_propagation = x_propagation.to(self.device)
        x_knowledge = x_knowledge.to(self.device)
        x_propagation_node_indices = [data.edge_index,data.x_propagation_node_indices_1_edge_index,data.x_propagation_node_indices_2_edge_index]
        x_propagation_node_values = [data.edge_values,data.x_propagation_node_values_1,data.x_propagation_node_values_2]
        x_progation_edge_num = [data.edge_num,data.x_propagation_edge_num_1,data.x_propagation_edge_num_2]

        x_knowledge_node_indices = [data.x_knowledge_node_indices_0_edge_index, data.x_knowledge_node_indices_1_edge_index,
                                      data.x_knowledge_node_indices_2_edge_index]
        x_knowledge_node_values = [data.x_knowledge_node_values_0, data.x_knowledge_node_values_1,
                                     data.x_knowledge_node_values_2]
        x_knowledge_edge_num = [data.x_knowledge_edge_num_0, data.x_knowledge_edge_num_1,
                                data.x_knowledge_edge_num_2]

        # batch_size = max(data.batch) + 1
        # print('batch_size', batch_size)
        # graph_dim = torch.sum(data.x_knowledge_node_num)
        # batch_k = torch.zeros(graph_dim).to(self.device)
        # temp_num = 0
        # for num_batch in range(batch_size):
        #     index = [idx for idx in range(temp_num,temp_num+data.x_knowledge_node_num[num_batch])]
        #     # print('index',index)
        #     temp_num += data.x_knowledge_node_num[num_batch]
        #     batch_k[index] = num_batch
        # print('batch_k',batch_k)

        # r = []
        # TODO: dynamic_graph那一篇paper 代码和文中所提不太一样
        # last_x = x_propagation
        for i, gcn_layer in enumerate(self.layer_stack_propagation):
            last_x = x_propagation
            x_gcn = gcn_layer(x_propagation, x_propagation_node_indices[i], x_propagation_node_values[i], root_idx_p, data.x_propagation_node_num, x_progation_edge_num[i],data.batch)
            # print('x_gcn:',x_gcn.size())
            x_propagation = self.bn_stack_propagation[i](x_gcn)
            x_knowledge = self.bn_stack_knowledge[i](self.layer_stack_knowledge[i](x_knowledge,x_knowledge_node_indices[i],x_knowledge_node_values[i],data.x_knowledge_node_num,x_knowledge_edge_num[i]))
            # if i < len(self.layer_stack) - 1:
            #====================================
            # x_propagation = F.relu(x_propagation)
            # x_knowledge = F.relu(x_knowledge)

            # x_propagation = F.dropout(x_propagation, self.dropout, training=self.training)
            # x_knowledge = F.dropout(x_knowledge,self.dropout,training = self.training)
            #====================================
            # print('attention test models:')
            # print('x_propagation size:',x_propagation.size())
            # print('x_knowledge size:',x_knowledge.size())
            x_propagation, x_knowledge = self.temporal_cells[i](x_propagation, x_knowledge,dual_graph_mutual_index_p,dual_graph_mutual_index_k,last_x,data.x_propagation_node_num,data.x_knowledge_node_num)  # temporal encoding
        #=========================================
        # x_propagation = F.relu(x_propagation)
        #===========================================
        # x_knowledge = F.relu(x_knowledge)
        # x = torch.cat([x_propagation,x_knowledge],1)
        # print('---model x propagation---:', x_propagation.size())
        x = self.mean(x_propagation, data.batch)
        x = x.squeeze(1)
        # print('model x size:',x.size())
        #TODO: x,text_encoding normalized(?)
        x_fusion = torch.cat([x,text_encoding_bert],dim=1)
        # print('---model x ---:', x.size()) #[32,128]
        x_fusion = F.leaky_relu(self.fc_rumor_1(x_fusion))
        x_fusion = self.fc_rumor_2(x_fusion)
        # x = self.mask(x, vertices)
        # print('----result before sigmoid----:',x)
        x_fusion = torch.sigmoid(x_fusion)
        # print('----result after sigmoid----:',x,x.size())
        return x_fusion
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#对propagation和knowledge图都进行了root—enhance||fusion模块中knowledge图只对post节点拼接，对全部节点进行变换，last_x为初始
class dual_DynamicGCN_nBatch_completed_1(nn.Module):
    def __init__(self, token_dict, mid2bert_tokenizer, bert_path, n_output,config,device, n_hidden=3, dropout=0.2, instance_norm=False):
        super(dual_DynamicGCN_nBatch_completed_1, self).__init__()
        self.token_dict = token_dict
        self.mid2bert_tokenizer = mid2bert_tokenizer
        self.config = config
        self.dropout = dropout
        self.instance_norm = instance_norm
        if self.instance_norm:
            self.norm = nn.InstanceNorm1d(self.config.graph_embedding_dim, momentum=0.0, affine=True)
        self.n_output = n_output
        self.device = device
        self.text_length = torch.tensor([self.config.text_max_length]).to(self.device)

        #-----------------------text-----------------------------------
        self.embedding = nn.Embedding(len(self.token_dict), config.graph_embedding_dim)
        self.gru = GRUEncoder(self.config.graph_embedding_dim, self.config.hidden_dim)
        # self.embedding.weight = nn.Parameter(pretrained_emb)
        # self.embedding.weight.requires_grad = False

        self.n_feature = self.config.hidden_dim
        # self.n_input = pretrained_emb.size(0)

        #------------------ ----bert-----------------------------------
        modelConfig = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path, config=modelConfig)
        for param in self.bert.parameters():
            param.requires_grad = True
        bert_outputdim = self.bert.config.hidden_size
        self.fc = nn.Linear(bert_outputdim, 128)

        #------------------------temporal GCN--------------------------
        self.layer_stack_propagation = nn.ModuleList()  # TODO class initiate
        self.layer_stack_knowledge = nn.ModuleList()
        self.bn_stack_propagation = nn.ModuleList()
        self.bn_stack_knowledge = nn.ModuleList()
        self.temporal_cells = nn.ModuleList()
        for i in range(n_hidden):
            self.layer_stack_propagation.append(PropagationGraphConvLayer_nBatch_geometric(self.n_feature, self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_propagation.append(nn.BatchNorm1d(self.n_feature))
            self.layer_stack_knowledge.append(KnowledgeGraphConvLayer_nBatch_geometric_1(self.n_feature,self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_knowledge.append(nn.BatchNorm1d(self.n_feature))
            self.temporal_cells.append(TemporalEncoding_nBatch_geometric_1(self.n_feature, self.n_feature))


        self.mean = Mean_nBatch_geometric(self.device,self.n_feature,self.n_feature)
        # self.linear = nn.Linear(self.n_feature,self.n_output)
        self.fc_rumor_1 = nn.Linear(256,100)
        self.fc_rumor_2 = nn.Linear(100,2)

    def forward(self, data):
        idx2encoding = {}
        features_propagation = []
        features_knowledge = []
        # text_encoding_bert = []

        #将图节点tensor转换为list
        idx_propagation_list = data.x.squeeze(1).cpu().numpy().tolist()
        idx_knowledge_list = data.x_knowledge_idx.squeeze(1).cpu().numpy().tolist()
        # print('---model idx_propagation_list---:', idx_propagation_list)

        #将root_idx对应node2idx转换为在图中的
        root_idx_p = []
        root_idx_k = []
        for i,root_idx in enumerate(data.root_idx):
            root_idx_item = root_idx.item()
            # bert
            input_ids = self.mid2bert_tokenizer[root_idx_item]['input_ids'].to(self.device)
            attention_mask_bert = self.mid2bert_tokenizer[root_idx_item]['attention_mask'].to(self.device)
            text_encoding = self.bert(input_ids,attention_mask=attention_mask_bert)
            text_encoding = text_encoding[0]
            text_encoding = torch.mean(text_encoding, dim=1,keepdim=True)
            text_encoding = text_encoding.squeeze(1)
            text_encoding = self.fc(text_encoding)
            # print('models encoding:',text_encoding) #1,1,768
            if i == 0:
                text_encoding_bert = text_encoding
            else:
                text_encoding_bert = torch.cat([text_encoding_bert,text_encoding],dim=0)

            root_idx_p.append(idx_propagation_list.index(root_idx_item))
            root_idx_k.append(idx_knowledge_list.index(root_idx_item))
        root_idx_p = torch.tensor(root_idx_p)
        root_idx_k = torch.tensor(root_idx_k)
        # print('---model root_idx_p---:',root_idx_p)


        for idx in data.x:
            idx = idx.item()
            text_embedding = self.embedding(torch.LongTensor([idx]).to(self.device))
            # text_embedding = torch.unsqueeze(text_embedding,dim=1)
            features_propagation.append(text_embedding.detach().cpu().numpy())


        # text_encoding_all = torch.tensor(text_encoding_bert)
        # print('models encoding_all:', text_encoding_bert.size())#8,128

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>comment>>>>>>>>>>>>>>>>>>>
        # for idx in
        dual_graph_mutual_index_p = []
        dual_graph_mutual_index_k = []
        for idx in data.x_knowledge_idx:
            idx = idx.item()
            token_idx = torch.LongTensor([idx]).to(self.device)
            entity_embedding = self.embedding(token_idx)
            features_knowledge.append(entity_embedding.detach().cpu().numpy())
            if idx in self.mid2bert_tokenizer:
                dual_graph_mutual_index_p.append(idx_propagation_list.index(idx))
                dual_graph_mutual_index_k.append(idx_knowledge_list.index(idx))
                # features_knowledge.append(idx2encoding[idx].detach().cpu().numpy())
        # print('model dual_graph_mutual_index_p:',dual_graph_mutual_index_p)
        # print('model dual_graph_mutual_index_k:',dual_graph_mutual_index_k)
        features_propagation = torch.tensor(features_propagation)
        features_knowledge = torch.tensor(features_knowledge)

        # emb_propagation = self.embedding(vertices_propagation)
        if self.instance_norm:
            features_propagation = self.norm(features_propagation)

        # emb_knowledge = self.embedding(vertices_knowledge)
        if self.instance_norm:
            features_knowledge = self.norm(features_knowledge)

        x_propagation = features_propagation
        x_knowledge = features_knowledge
        # print('x_propagation:',features_propagation.size()) #7,1,128
        # print('x_knowleddge:',features_knowledge.size()) #771,1,128
        x_propagation = torch.squeeze(x_propagation,dim=1)
        x_knowledge = torch.squeeze(x_knowledge,dim=1)
        # print('x_propagation:', x_propagation.size())  # 7,1,128
        # print('x_knowleddge:', x_knowledge.size())  # 771,1,128
        # print('adj_propagation:',adjs_propagation[0].size())
        # print('adj_knowledge:',adjs_knowledge[0].size())

        x_propagation = x_propagation.to(self.device)
        x_knowledge = x_knowledge.to(self.device)
        x_propagation_node_indices = [data.edge_index,data.x_propagation_node_indices_1_edge_index,data.x_propagation_node_indices_2_edge_index]
        x_propagation_node_values = [data.edge_values,data.x_propagation_node_values_1,data.x_propagation_node_values_2]
        x_progation_edge_num = [data.edge_num,data.x_propagation_edge_num_1,data.x_propagation_edge_num_2]

        x_knowledge_node_indices = [data.x_knowledge_node_indices_0_edge_index, data.x_knowledge_node_indices_1_edge_index,
                                      data.x_knowledge_node_indices_2_edge_index]
        x_knowledge_node_values = [data.x_knowledge_node_values_0, data.x_knowledge_node_values_1,
                                     data.x_knowledge_node_values_2]
        x_knowledge_edge_num = [data.x_knowledge_edge_num_0, data.x_knowledge_edge_num_1,
                                data.x_knowledge_edge_num_2]
        # r = []
        # TODO: dynamic_graph那一篇paper 代码和文中所提不太一样
        last_x = x_knowledge
        for i, gcn_layer in enumerate(self.layer_stack_propagation):
            # last_x = x_propagation
            x_gcn = gcn_layer(x_propagation, x_propagation_node_indices[i], x_propagation_node_values[i], root_idx_p, data.x_propagation_node_num, x_progation_edge_num[i],data.batch)
            # print('x_gcn:',x_gcn.size())
            x_propagation = self.bn_stack_propagation[i](x_gcn)
            x_knowledge = self.bn_stack_knowledge[i](self.layer_stack_knowledge[i](x_knowledge,x_knowledge_node_indices[i],x_knowledge_node_values[i], root_idx_k,data.x_knowledge_node_num,x_knowledge_edge_num[i],data.batch))
            # if i < len(self.layer_stack) - 1:
            #====================================
            x_propagation = F.leaky_relu(x_propagation)
            x_knowledge = F.leaky_relu(x_knowledge)

            x_propagation = F.dropout(x_propagation, self.dropout, training=self.training)
            x_knowledge = F.dropout(x_knowledge,self.dropout,training = self.training)
            #====================================
            # print('attention test models:')
            # print('x_propagation size:',x_propagation.size())
            # print('x_knowledge size:',x_knowledge.size())
            x_propagation, x_knowledge = self.temporal_cells[i](x_propagation, x_knowledge,dual_graph_mutual_index_p,dual_graph_mutual_index_k,last_x,data.x_propagation_node_num,data.x_knowledge_node_num)  # temporal encoding
        #=========================================
        # x_propagation = F.relu(x_propagation)
        #===========================================
        # x_knowledge = F.relu(x_knowledge)
        # x = torch.cat([x_propagation,x_knowledge],1)
        # print('---model x propagation---:', x_propagation.size())
        x = self.mean(x_propagation, data.batch)
        x = x.squeeze(1)
        # print('model x size:',x.size())
        #TODO: x,text_encoding normalized(?)
        x_fusion = torch.cat([x,text_encoding_bert],dim=1)
        # print('---model x ---:', x.size()) #[32,128]
        x_fusion = F.leaky_relu(self.fc_rumor_1(x_fusion))
        x_fusion = self.fc_rumor_2(x_fusion)
        # x = self.mask(x, vertices)
        # print('----result before sigmoid----:',x)
        x_fusion = torch.sigmoid(x_fusion)
        # print('----result after sigmoid----:',x,x.size())
        return x_fusion
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#只对propagation图进行了root—enhance||fusion模块中knowledge图对全部节点进行拼接，类RNN，last_x作为上一stage的hidden
class dual_DynamicGCN_nBatch_completed_2(nn.Module):
    def __init__(self, token_dict, mid2bert_tokenizer, bert_path, n_output,config,device, n_hidden=3, dropout=0.2, instance_norm=False):
        super(dual_DynamicGCN_nBatch_completed_2, self).__init__()
        self.token_dict = token_dict
        self.mid2bert_tokenizer = mid2bert_tokenizer
        self.config = config
        self.dropout = dropout
        self.instance_norm = instance_norm
        if self.instance_norm:
            self.norm = nn.InstanceNorm1d(self.config.graph_embedding_dim, momentum=0.0, affine=True)
        self.n_output = n_output
        self.device = device
        self.text_length = torch.tensor([self.config.text_max_length]).to(self.device)

        #-----------------------text-----------------------------------
        self.embedding = nn.Embedding(len(self.token_dict), config.graph_embedding_dim)
        self.gru = GRUEncoder(self.config.graph_embedding_dim, self.config.hidden_dim)
        # self.embedding.weight = nn.Parameter(pretrained_emb)
        # self.embedding.weight.requires_grad = False

        self.n_feature = self.config.hidden_dim
        # self.n_input = pretrained_emb.size(0)

        #------------------ ----bert-----------------------------------
        modelConfig = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path, config=modelConfig)
        for param in self.bert.parameters():
            param.requires_grad = True
        bert_outputdim = self.bert.config.hidden_size
        self.fc = nn.Linear(bert_outputdim, 128)

        #------------------------temporal GCN--------------------------
        self.layer_stack_propagation = nn.ModuleList()  # TODO class initiate
        self.layer_stack_knowledge = nn.ModuleList()
        self.bn_stack_propagation = nn.ModuleList()
        self.bn_stack_knowledge = nn.ModuleList()
        self.temporal_cells = nn.ModuleList()
        for i in range(n_hidden):
            self.layer_stack_propagation.append(PropagationGraphConvLayer_nBatch_geometric(self.n_feature, self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_propagation.append(nn.BatchNorm1d(self.n_feature))
            self.layer_stack_knowledge.append(KnowledgeGraphConvLayer_nBatch_geometric(self.n_feature,self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_knowledge.append(nn.BatchNorm1d(self.n_feature))
            self.temporal_cells.append(TemporalEncoding_nBatch_geometric_2(self.n_feature, self.n_feature))


        self.mean = Mean_nBatch_geometric(self.device,self.n_feature,self.n_feature)
        # self.linear = nn.Linear(self.n_feature,self.n_output)
        self.fc_rumor_1 = nn.Linear(256,100)
        self.fc_rumor_2 = nn.Linear(100,2)

    def forward(self, data):
        idx2encoding = {}
        features_propagation = []
        features_knowledge = []
        # text_encoding_bert = []

        #将图节点tensor转换为list
        idx_propagation_list = data.x.squeeze(1).cpu().numpy().tolist()
        idx_knowledge_list = data.x_knowledge_idx.squeeze(1).cpu().numpy().tolist()
        # print('---model idx_propagation_list---:', idx_propagation_list)

        #将root_idx对应node2idx转换为在图中的
        root_idx_p = []
        root_idx_k = []
        for i,root_idx in enumerate(data.root_idx):
            root_idx_item = root_idx.item()
            # bert
            input_ids = self.mid2bert_tokenizer[root_idx_item]['input_ids'].to(self.device)
            attention_mask_bert = self.mid2bert_tokenizer[root_idx_item]['attention_mask'].to(self.device)
            text_encoding = self.bert(input_ids,attention_mask=attention_mask_bert)
            text_encoding = text_encoding[0]
            text_encoding = torch.mean(text_encoding, dim=1,keepdim=True)
            text_encoding = text_encoding.squeeze(1)
            text_encoding = self.fc(text_encoding)
            # print('models encoding:',text_encoding) #1,1,768
            if i == 0:
                text_encoding_bert = text_encoding
            else:
                text_encoding_bert = torch.cat([text_encoding_bert,text_encoding],dim=0)

            root_idx_p.append(idx_propagation_list.index(root_idx_item))
            root_idx_k.append(idx_knowledge_list.index(root_idx_item))
        root_idx_p = torch.tensor(root_idx_p)
        root_idx_k = torch.tensor(root_idx_k)
        # print('---model root_idx_p---:',root_idx_p)


        for idx in data.x:
            idx = idx.item()
            text_embedding = self.embedding(torch.LongTensor([idx]).to(self.device))
            # text_embedding = torch.unsqueeze(text_embedding,dim=1)
            features_propagation.append(text_embedding.detach().cpu().numpy())


        # text_encoding_all = torch.tensor(text_encoding_bert)
        # print('models encoding_all:', text_encoding_bert.size())#8,128

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>comment>>>>>>>>>>>>>>>>>>>
        # for idx in
        dual_graph_mutual_index_p = []
        dual_graph_mutual_index_k = []
        for idx in data.x_knowledge_idx:
            idx = idx.item()
            token_idx = torch.LongTensor([idx]).to(self.device)
            entity_embedding = self.embedding(token_idx)
            features_knowledge.append(entity_embedding.detach().cpu().numpy())
            if idx in self.mid2bert_tokenizer:
                dual_graph_mutual_index_p.append(idx_propagation_list.index(idx))
                dual_graph_mutual_index_k.append(idx_knowledge_list.index(idx))
                # features_knowledge.append(idx2encoding[idx].detach().cpu().numpy())
        # print('model dual_graph_mutual_index_p:',dual_graph_mutual_index_p)
        # print('model dual_graph_mutual_index_k:',dual_graph_mutual_index_k)
        features_propagation = torch.tensor(features_propagation)
        features_knowledge = torch.tensor(features_knowledge)

        # emb_propagation = self.embedding(vertices_propagation)
        if self.instance_norm:
            features_propagation = self.norm(features_propagation)

        # emb_knowledge = self.embedding(vertices_knowledge)
        if self.instance_norm:
            features_knowledge = self.norm(features_knowledge)

        x_propagation = features_propagation
        x_knowledge = features_knowledge
        # print('x_propagation:',features_propagation.size()) #7,1,128
        # print('x_knowleddge:',features_knowledge.size()) #771,1,128
        x_propagation = torch.squeeze(x_propagation,dim=1)
        x_knowledge = torch.squeeze(x_knowledge,dim=1)
        # print('x_propagation:', x_propagation.size())  # 7,1,128
        # print('x_knowleddge:', x_knowledge.size())  # 771,1,128
        # print('adj_propagation:',adjs_propagation[0].size())
        # print('adj_knowledge:',adjs_knowledge[0].size())

        x_propagation = x_propagation.to(self.device)
        x_knowledge = x_knowledge.to(self.device)
        x_propagation_node_indices = [data.edge_index,data.x_propagation_node_indices_1_edge_index,data.x_propagation_node_indices_2_edge_index]
        x_propagation_node_values = [data.edge_values,data.x_propagation_node_values_1,data.x_propagation_node_values_2]
        x_progation_edge_num = [data.edge_num,data.x_propagation_edge_num_1,data.x_propagation_edge_num_2]

        x_knowledge_node_indices = [data.x_knowledge_node_indices_0_edge_index, data.x_knowledge_node_indices_1_edge_index,
                                      data.x_knowledge_node_indices_2_edge_index]
        x_knowledge_node_values = [data.x_knowledge_node_values_0, data.x_knowledge_node_values_1,
                                     data.x_knowledge_node_values_2]
        x_knowledge_edge_num = [data.x_knowledge_edge_num_0, data.x_knowledge_edge_num_1,
                                data.x_knowledge_edge_num_2]
        # r = []
        # TODO: dynamic_graph那一篇paper 代码和文中所提不太一样

        for i, gcn_layer in enumerate(self.layer_stack_propagation):
            # last_x = x_propagation
            last_x = x_knowledge
            x_gcn = gcn_layer(x_propagation, x_propagation_node_indices[i], x_propagation_node_values[i], root_idx_p, data.x_propagation_node_num, x_progation_edge_num[i],data.batch)
            # print('x_gcn:',x_gcn.size())
            x_propagation = self.bn_stack_propagation[i](x_gcn)
            x_knowledge = self.bn_stack_knowledge[i](self.layer_stack_knowledge[i](x_knowledge,x_knowledge_node_indices[i],x_knowledge_node_values[i], data.x_knowledge_node_num,x_knowledge_edge_num[i]))
            # if i < len(self.layer_stack) - 1:
            #====================================
            x_propagation = F.leaky_relu(x_propagation)
            x_knowledge = F.leaky_relu(x_knowledge)

            x_propagation = F.dropout(x_propagation, self.dropout, training=self.training)
            x_knowledge = F.dropout(x_knowledge,self.dropout,training = self.training)
            #====================================
            # print('attention test models:')
            # print('x_propagation size:',x_propagation.size())
            # print('x_knowledge size:',x_knowledge.size())
            x_propagation, x_knowledge = self.temporal_cells[i](x_propagation, x_knowledge,dual_graph_mutual_index_p,dual_graph_mutual_index_k,last_x,data.x_propagation_node_num,data.x_knowledge_node_num)  # temporal encoding
        #=========================================
        # x_propagation = F.relu(x_propagation)
        #===========================================
        # x_knowledge = F.relu(x_knowledge)
        # x = torch.cat([x_propagation,x_knowledge],1)
        # print('---model x propagation---:', x_propagation.size())
        x = self.mean(x_propagation, data.batch)
        x = x.squeeze(1)
        # print('model x size:',x.size())
        #TODO: x,text_encoding normalized(?)
        x_fusion = torch.cat([x,text_encoding_bert],dim=1)
        # print('---model x ---:', x.size()) #[32,128]
        x_fusion = F.leaky_relu(self.fc_rumor_1(x_fusion))
        x_fusion = self.fc_rumor_2(x_fusion)
        # x = self.mask(x, vertices)
        # print('----result before sigmoid----:',x)
        x_fusion = torch.sigmoid(x_fusion)
        # print('----result after sigmoid----:',x,x.size())
        return x_fusion
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


class dual_DynamicGCN_nBatch_completed_wobert(nn.Module):
    def __init__(self, token_dict, mid2bert_tokenizer, bert_path, n_output,config,device, n_hidden=3, dropout=0.2, instance_norm=False):
        super(dual_DynamicGCN_nBatch_completed_wobert, self).__init__()
        self.token_dict = token_dict
        self.mid2bert_tokenizer = mid2bert_tokenizer
        self.config = config
        self.dropout = dropout
        self.instance_norm = instance_norm
        if self.instance_norm:
            self.norm = nn.InstanceNorm1d(self.config.graph_embedding_dim, momentum=0.0, affine=True)
        self.n_output = n_output
        self.device = device
        self.text_length = torch.tensor([self.config.text_max_length]).to(self.device)

        #-----------------------text-----------------------------------
        self.embedding = nn.Embedding(len(self.token_dict), config.graph_embedding_dim)
        self.gru = GRUEncoder(self.config.graph_embedding_dim, self.config.hidden_dim)
        # self.embedding.weight = nn.Parameter(pretrained_emb)
        # self.embedding.weight.requires_grad = False

        self.n_feature = self.config.hidden_dim
        # self.n_input = pretrained_emb.size(0)

        #------------------ ----bert-----------------------------------
        modelConfig = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path, config=modelConfig)
        for param in self.bert.parameters():
            param.requires_grad = True
        bert_outputdim = self.bert.config.hidden_size
        self.fc = nn.Linear(bert_outputdim, 128)


        #------------------------temporal GCN--------------------------
        self.layer_stack_propagation = nn.ModuleList()  # TODO class initiate
        self.layer_stack_knowledge = nn.ModuleList()
        self.bn_stack_propagation = nn.ModuleList()
        self.bn_stack_knowledge = nn.ModuleList()
        self.temporal_cells = nn.ModuleList()
        for i in range(n_hidden):
            self.layer_stack_propagation.append(PropagationGraphConvLayer_nBatch_geometric(self.n_feature, self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_propagation.append(nn.BatchNorm1d(self.n_feature))
            self.layer_stack_knowledge.append(KnowledgeGraphConvLayer_nBatch_geometric(self.n_feature,self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_knowledge.append(nn.BatchNorm1d(self.n_feature))
            self.temporal_cells.append(TemporalEncoding_nBatch_geometric(self.n_feature, self.n_feature))


        self.mean = Mean_nBatch_geometric(self.device,self.n_feature,self.n_feature)
        # self.linear = nn.Linear(self.n_feature,self.n_output)
        self.fc_rumor_1 = nn.Linear(128,100)
        self.fc_rumor_2 = nn.Linear(100,2)

    def forward(self, data):
        idx2encoding = {}
        features_propagation = []
        features_knowledge = []
        # text_encoding_bert = []

        #将图节点tensor转换为list
        idx_propagation_list = data.x.squeeze(1).cpu().numpy().tolist()
        idx_knowledge_list = data.x_knowledge_idx.squeeze(1).cpu().numpy().tolist()
        # print('---model idx_propagation_list---:', idx_propagation_list)

        #将root_idx对应node2idx转换为在图中的
        root_idx_p = []
        for i,root_idx in enumerate(data.root_idx):
            root_idx_item = root_idx.item()
            # bert
            input_ids = self.mid2bert_tokenizer[root_idx_item]['input_ids'].to(self.device)
            attention_mask_bert = self.mid2bert_tokenizer[root_idx_item]['attention_mask'].to(self.device)
            text_encoding = self.bert(input_ids,attention_mask=attention_mask_bert)
            text_encoding = text_encoding[0]
            text_encoding = torch.mean(text_encoding, dim=1,keepdim=True)
            text_encoding = text_encoding.squeeze(1)
            text_encoding = self.fc(text_encoding)
            # print('models encoding:',text_encoding) #1,1,768
            if i == 0:
                text_encoding_bert = text_encoding
            else:
                text_encoding_bert = torch.cat([text_encoding_bert,text_encoding],dim=0)

            root_idx_p.append(idx_propagation_list.index(root_idx_item))
        root_idx_p = torch.tensor(root_idx_p)
        # print('---model root_idx_p---:',root_idx_p)


        for idx in data.x:
            idx = idx.item()
            text_embedding = self.embedding(torch.LongTensor([idx]).to(self.device))
            # text_embedding = torch.unsqueeze(text_embedding,dim=1)
            features_propagation.append(text_embedding.detach().cpu().numpy())


        # text_encoding_all = torch.tensor(text_encoding_bert)
        # print('models encoding_all:', text_encoding_bert.size())#8,128

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>comment>>>>>>>>>>>>>>>>>>>
        # for idx in
        dual_graph_mutual_index_p = []
        dual_graph_mutual_index_k = []
        for idx in data.x_knowledge_idx:
            idx = idx.item()
            token_idx = torch.LongTensor([idx]).to(self.device)
            entity_embedding = self.embedding(token_idx)
            features_knowledge.append(entity_embedding.detach().cpu().numpy())
            if idx in self.mid2bert_tokenizer:
                dual_graph_mutual_index_p.append(idx_propagation_list.index(idx))
                dual_graph_mutual_index_k.append(idx_knowledge_list.index(idx))
                # features_knowledge.append(idx2encoding[idx].detach().cpu().numpy())
        # print('model dual_graph_mutual_index_p:',dual_graph_mutual_index_p)
        # print('model dual_graph_mutual_index_k:',dual_graph_mutual_index_k)
        features_propagation = torch.tensor(features_propagation)
        features_knowledge = torch.tensor(features_knowledge)

        # emb_propagation = self.embedding(vertices_propagation)
        if self.instance_norm:
            features_propagation = self.norm(features_propagation)

        # emb_knowledge = self.embedding(vertices_knowledge)
        if self.instance_norm:
            features_knowledge = self.norm(features_knowledge)

        x_propagation = features_propagation
        x_knowledge = features_knowledge
        # print('x_propagation:',features_propagation.size()) #7,1,128
        # print('x_knowleddge:',features_knowledge.size()) #771,1,128
        x_propagation = torch.squeeze(x_propagation,dim=1)
        x_knowledge = torch.squeeze(x_knowledge,dim=1)
        # print('x_propagation:', x_propagation.size())  # 7,1,128
        # print('x_knowleddge:', x_knowledge.size())  # 771,1,128
        # print('adj_propagation:',adjs_propagation[0].size())
        # print('adj_knowledge:',adjs_knowledge[0].size())

        x_propagation = x_propagation.to(self.device)
        x_knowledge = x_knowledge.to(self.device)
        # print('model x_propagation:',x_propagation)
        # print('model x_knowledge:',x_knowledge)

        x_propagation_node_indices = [data.edge_index,data.x_propagation_node_indices_1_edge_index,data.x_propagation_node_indices_2_edge_index]
        x_propagation_node_values = [data.edge_values,data.x_propagation_node_values_1,data.x_propagation_node_values_2]
        x_progation_edge_num = [data.edge_num,data.x_propagation_edge_num_1,data.x_propagation_edge_num_2]

        x_knowledge_node_indices = [data.x_knowledge_node_indices_0_edge_index, data.x_knowledge_node_indices_1_edge_index,
                                      data.x_knowledge_node_indices_2_edge_index]
        x_knowledge_node_values = [data.x_knowledge_node_values_0, data.x_knowledge_node_values_1,
                                     data.x_knowledge_node_values_2]
        x_knowledge_edge_num = [data.x_knowledge_edge_num_0, data.x_knowledge_edge_num_1,
                                data.x_knowledge_edge_num_2]
        # r = []

        # TODO: dynamic_graph那一篇paper 代码和文中所提不太一样

        for i, gcn_layer in enumerate(self.layer_stack_propagation):
            last_x = x_propagation
            x_gcn = gcn_layer(x_propagation, x_propagation_node_indices[i], x_propagation_node_values[i], root_idx_p, data.x_propagation_node_num, x_progation_edge_num[i],data.batch)
            # print('x_gcn:',x_gcn.size())
        # ========================================================================
            x_propagation = self.bn_stack_propagation[i](x_gcn)
            x_knowledge = self.bn_stack_knowledge[i](self.layer_stack_knowledge[i](x_knowledge,x_knowledge_node_indices[i],x_knowledge_node_values[i],data.x_knowledge_node_num,x_knowledge_edge_num[i]))
            # if i < len(self.layer_stack) - 1:
            #=========
            # x_propagation = F.leaky_relu(x_propagation)
            # x_knowledge = F.leaky_relu(x_knowledge)

            # x_propagation = F.dropout(x_propagation, self.dropout, training=self.training)
            # x_knowledge = F.dropout(x_knowledge,self.dropout,training = self.training)
            #==========
            # print('attention test models:')
            # print('x_propagation size:',x_propagation.size())
            # print('x_knowledge size:',x_knowledge.size())
            x_propagation, x_knowledge = self.temporal_cells[i](x_propagation, x_knowledge,dual_graph_mutual_index_p,dual_graph_mutual_index_k,last_x,data.x_propagation_node_num,data.x_knowledge_node_num)  # temporal encoding

        #todo 去掉了一个leaky_rulu
        # x_propagation = F.leaky_relu(x_propagation)
        # x_knowledge = F.relu(x_knowledge)
        # x = torch.cat([x_propagation,x_knowledge],1)
        # print('---model x propagation---:', x_propagation.size())
        x = self.mean(x_propagation, data.batch)
        x = x.squeeze(1)
        # print('model x size:',x.size())
        #TODO: x,text_encoding normalized(?)
        # x_fusion = torch.cat([x,text_encoding_bert],dim=1)
        # print('---model x ---:', x.size()) #[32,128]
        x_fusion = F.leaky_relu(self.fc_rumor_1(x))
        x_fusion = self.fc_rumor_2(x_fusion)
        # x = self.mask(x, vertices)
        # print('----result before sigmoid----:',x)
        x_fusion = torch.sigmoid(x_fusion)
        # print('----result after sigmoid----:',x,x.size())
        return x_fusion
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#temporal fusion换为attention融合
class dual_DynamicGCN_nBatch_completed_attentionTF(nn.Module):
    def __init__(self, token_dict, mid2bert_tokenizer, bert_path, n_output,config,device, n_hidden=3, dropout=0.2, instance_norm=False):
        super(dual_DynamicGCN_nBatch_completed_attentionTF, self).__init__()
        self.token_dict = token_dict
        self.mid2bert_tokenizer = mid2bert_tokenizer
        self.config = config
        self.dropout = dropout
        self.instance_norm = instance_norm
        if self.instance_norm:
            self.norm = nn.InstanceNorm1d(self.config.graph_embedding_dim, momentum=0.0, affine=True)
        self.n_output = n_output
        self.device = device
        self.text_length = torch.tensor([self.config.text_max_length]).to(self.device)

        #-----------------------text-----------------------------------
        self.embedding = nn.Embedding(len(self.token_dict), config.graph_embedding_dim)
        self.gru = GRUEncoder(self.config.graph_embedding_dim, self.config.hidden_dim)
        # self.embedding.weight = nn.Parameter(pretrained_emb)
        # self.embedding.weight.requires_grad = False

        self.n_feature = self.config.hidden_dim
        # self.n_input = pretrained_emb.size(0)

        #------------------ ----bert-----------------------------------
        modelConfig = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path, config=modelConfig)
        for param in self.bert.parameters():
            param.requires_grad = True
        bert_outputdim = self.bert.config.hidden_size
        self.fc = nn.Linear(bert_outputdim, 128)


        #------------------------temporal GCN--------------------------
        self.layer_stack_propagation = nn.ModuleList()  # TODO class initiate
        self.layer_stack_knowledge = nn.ModuleList()
        self.bn_stack_propagation = nn.ModuleList()
        self.bn_stack_knowledge = nn.ModuleList()
        self.temporal_cells = nn.ModuleList()
        for i in range(n_hidden):
            self.layer_stack_propagation.append(PropagationGraphConvLayer_nBatch_geometric(self.n_feature, self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_propagation.append(nn.BatchNorm1d(self.n_feature))
            self.layer_stack_knowledge.append(KnowledgeGraphConvLayer_nBatch_geometric(self.n_feature,self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_knowledge.append(nn.BatchNorm1d(self.n_feature))
            self.temporal_cells.append(TemporalEncoding_nBatch_geometric_attention(self.n_feature, self.n_feature))


        self.mean = Mean_nBatch_geometric(self.device,self.n_feature,self.n_feature)
        # self.linear = nn.Linear(self.n_feature,self.n_output)
        self.fc_rumor_1 = nn.Linear(256,100)
        self.fc_rumor_2 = nn.Linear(100,2)

    def forward(self, data):
        idx2encoding = {}
        features_propagation = []
        features_knowledge = []
        # text_encoding_bert = []

        #将图节点tensor转换为list
        idx_propagation_list = data.x.squeeze(1).cpu().numpy().tolist()
        idx_knowledge_list = data.x_knowledge_idx.squeeze(1).cpu().numpy().tolist()
        # print('---model idx_propagation_list---:', idx_propagation_list)

        #将root_idx对应node2idx转换为在图中的
        root_idx_p = []
        for i,root_idx in enumerate(data.root_idx):
            root_idx_item = root_idx.item()
            # bert
            input_ids = self.mid2bert_tokenizer[root_idx_item]['input_ids'].to(self.device)
            attention_mask_bert = self.mid2bert_tokenizer[root_idx_item]['attention_mask'].to(self.device)
            text_encoding = self.bert(input_ids,attention_mask=attention_mask_bert)
            text_encoding = text_encoding[0]
            text_encoding = torch.mean(text_encoding, dim=1,keepdim=True)
            text_encoding = text_encoding.squeeze(1)
            text_encoding = self.fc(text_encoding)
            # print('models encoding:',text_encoding) #1,1,768
            if i == 0:
                text_encoding_bert = text_encoding
            else:
                text_encoding_bert = torch.cat([text_encoding_bert,text_encoding],dim=0)

            root_idx_p.append(idx_propagation_list.index(root_idx_item))
        root_idx_p = torch.tensor(root_idx_p)
        # print('---model root_idx_p---:',root_idx_p)


        for idx in data.x:
            idx = idx.item()
            text_embedding = self.embedding(torch.LongTensor([idx]).to(self.device))
            # text_embedding = torch.unsqueeze(text_embedding,dim=1)
            features_propagation.append(text_embedding.detach().cpu().numpy())


        # text_encoding_all = torch.tensor(text_encoding_bert)
        # print('models encoding_all:', text_encoding_bert.size())#8,128

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>comment>>>>>>>>>>>>>>>>>>>
        # for idx in
        dual_graph_mutual_index_p = []
        dual_graph_mutual_index_k = []
        for idx in data.x_knowledge_idx:
            idx = idx.item()
            token_idx = torch.LongTensor([idx]).to(self.device)
            entity_embedding = self.embedding(token_idx)
            features_knowledge.append(entity_embedding.detach().cpu().numpy())
            if idx in self.mid2bert_tokenizer:
                dual_graph_mutual_index_p.append(idx_propagation_list.index(idx))
                dual_graph_mutual_index_k.append(idx_knowledge_list.index(idx))
                # features_knowledge.append(idx2encoding[idx].detach().cpu().numpy())
        # print('model dual_graph_mutual_index_p:',dual_graph_mutual_index_p)
        # print('model dual_graph_mutual_index_k:',dual_graph_mutual_index_k)
        features_propagation = torch.tensor(features_propagation)
        features_knowledge = torch.tensor(features_knowledge)

        # emb_propagation = self.embedding(vertices_propagation)
        if self.instance_norm:
            features_propagation = self.norm(features_propagation)

        # emb_knowledge = self.embedding(vertices_knowledge)
        if self.instance_norm:
            features_knowledge = self.norm(features_knowledge)

        x_propagation = features_propagation
        x_knowledge = features_knowledge
        # print('x_propagation:',features_propagation.size()) #7,1,128
        # print('x_knowleddge:',features_knowledge.size()) #771,1,128
        x_propagation = torch.squeeze(x_propagation,dim=1)
        x_knowledge = torch.squeeze(x_knowledge,dim=1)
        # print('x_propagation:', x_propagation.size())  # 7,1,128
        # print('x_knowleddge:', x_knowledge.size())  # 771,1,128
        # print('adj_propagation:',adjs_propagation[0].size())
        # print('adj_knowledge:',adjs_knowledge[0].size())

        x_propagation = x_propagation.to(self.device)
        x_knowledge = x_knowledge.to(self.device)
        x_propagation_node_indices = [data.edge_index,data.x_propagation_node_indices_1_edge_index,data.x_propagation_node_indices_2_edge_index]
        x_propagation_node_values = [data.edge_values,data.x_propagation_node_values_1,data.x_propagation_node_values_2]
        x_progation_edge_num = [data.edge_num,data.x_propagation_edge_num_1,data.x_propagation_edge_num_2]

        x_knowledge_node_indices = [data.x_knowledge_node_indices_0_edge_index, data.x_knowledge_node_indices_1_edge_index,
                                      data.x_knowledge_node_indices_2_edge_index]
        x_knowledge_node_values = [data.x_knowledge_node_values_0, data.x_knowledge_node_values_1,
                                     data.x_knowledge_node_values_2]
        x_knowledge_edge_num = [data.x_knowledge_edge_num_0, data.x_knowledge_edge_num_1,
                                data.x_knowledge_edge_num_2]
        # r = []
        # TODO: dynamic_graph那一篇paper 代码和文中所提不太一样
        last_x = x_propagation
        for i, gcn_layer in enumerate(self.layer_stack_propagation):

            x_gcn = gcn_layer(x_propagation, x_propagation_node_indices[i], x_propagation_node_values[i], root_idx_p, data.x_propagation_node_num, x_progation_edge_num[i],data.batch)
            # print('x_gcn:',x_gcn.size())
            x_propagation = self.bn_stack_propagation[i](x_gcn)
            x_knowledge = self.bn_stack_knowledge[i](self.layer_stack_knowledge[i](x_knowledge,x_knowledge_node_indices[i],x_knowledge_node_values[i],data.x_knowledge_node_num,x_knowledge_edge_num[i]))
            # if i < len(self.layer_stack) - 1:
            x_propagation = F.relu(x_propagation)
            x_knowledge = F.relu(x_knowledge)

            x_propagation = F.dropout(x_propagation, self.dropout, training=self.training)
            x_knowledge = F.dropout(x_knowledge,self.dropout,training = self.training)

            # print('attention test models:')
            # print('x_propagation size:',x_propagation.size())
            # print('x_knowledge size:',x_knowledge.size())
            x_propagation, x_knowledge = self.temporal_cells[i](x_propagation, x_knowledge,dual_graph_mutual_index_p,dual_graph_mutual_index_k,last_x,data.x_propagation_node_num,data.x_knowledge_node_num)  # temporal encoding

        x_propagation = F.relu(x_propagation)
        # x_knowledge = F.relu(x_knowledge)
        # x = torch.cat([x_propagation,x_knowledge],1)
        # print('---model x propagation---:', x_propagation.size())
        x = self.mean(x_propagation, data.batch)
        x = x.squeeze(1)
        # print('model x size:',x.size())
        #TODO: x,text_encoding normalized(?)
        x_fusion = torch.cat([x,text_encoding_bert],dim=1)
        # print('---model x ---:', x.size()) #[32,128]
        x_fusion = F.relu(self.fc_rumor_1(x_fusion))
        x_fusion = self.fc_rumor_2(x_fusion)
        # x = self.mask(x, vertices)
        # print('----result before sigmoid----:',x)
        x_fusion = torch.sigmoid(x_fusion)
        # print('----result after sigmoid----:',x,x.size())
        return x_fusion
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#post节点的初始embedding换成gru,不加最后bert的信息
class dual_DynamicGCN_nBatch_completed_gru_wobert(nn.Module):
    def __init__(self, token_dict, mid2regtextidx_dict, bert_path, n_output,config,device, n_hidden=3, dropout=0.2, instance_norm=False):
        super(dual_DynamicGCN_nBatch_completed_gru_wobert, self).__init__()
        self.token_dict = token_dict
        self.mid2regtextidx_dict = mid2regtextidx_dict
        self.config = config
        self.dropout = dropout
        self.instance_norm = instance_norm
        if self.instance_norm:
            self.norm = nn.InstanceNorm1d(self.config.graph_embedding_dim, momentum=0.0, affine=True)
        self.n_output = n_output
        self.device = device
        self.text_length = torch.tensor([self.config.text_max_length]).to(self.device)

        #-----------------------text-----------------------------------
        self.embedding = nn.Embedding(len(self.token_dict), config.graph_embedding_dim)
        self.gru = GRUEncoder(self.config.graph_embedding_dim, self.config.hidden_dim)
        # self.embedding.weight = nn.Parameter(pretrained_emb)
        # self.embedding.weight.requires_grad = False

        self.n_feature = self.config.hidden_dim
        # self.n_input = pretrained_emb.size(0)

        #------------------ ----bert-----------------------------------
        # modelConfig = BertConfig.from_pretrained(bert_path)
        # self.bert = BertModel.from_pretrained(bert_path, config=modelConfig)
        # for param in self.bert.parameters():
        #     param.requires_grad = True
        # bert_outputdim = self.bert.config.hidden_size
        # self.fc = nn.Linear(bert_outputdim, 128)


        #------------------------temporal GCN--------------------------
        self.layer_stack_propagation = nn.ModuleList()  # TODO class initiate
        self.layer_stack_knowledge = nn.ModuleList()
        self.bn_stack_propagation = nn.ModuleList()
        self.bn_stack_knowledge = nn.ModuleList()
        self.temporal_cells = nn.ModuleList()
        for i in range(n_hidden):
            self.layer_stack_propagation.append(PropagationGraphConvLayer_nBatch_geometric(self.n_feature, self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_propagation.append(nn.BatchNorm1d(self.n_feature))
            self.layer_stack_knowledge.append(KnowledgeGraphConvLayer_nBatch_geometric(self.n_feature,self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_knowledge.append(nn.BatchNorm1d(self.n_feature))
            self.temporal_cells.append(TemporalEncoding_nBatch_geometric(self.n_feature, self.n_feature))


        self.mean = Mean_nBatch_geometric(self.device,self.n_feature,self.n_feature)
        # self.linear = nn.Linear(self.n_feature,self.n_output)
        self.fc_rumor_1 = nn.Linear(128,100)
        self.fc_rumor_2 = nn.Linear(100,2)

    def forward(self, data):
        idx2encoding = {}
        features_propagation = []
        features_knowledge = []
        # text_encoding_bert = []

        #将图节点tensor转换为list
        idx_propagation_list = data.x.squeeze(1).cpu().numpy().tolist()
        idx_knowledge_list = data.x_knowledge_idx.squeeze(1).cpu().numpy().tolist()
        # print('---model idx_propagation_list---:', idx_propagation_list)

        #将root_idx对应node2idx转换为在图中的
        root_idx_p = []
        for i,root_idx in enumerate(data.root_idx):
            root_idx_item = root_idx.item()
            # bert
            # input_ids = self.mid2regtextidx_dict[root_idx_item]['input_ids'].to(self.device)
            # attention_mask_bert = self.mid2regtextidx_dict[root_idx_item]['attention_mask'].to(self.device)
            # text_encoding = self.bert(input_ids,attention_mask=attention_mask_bert)
            # text_encoding = text_encoding[0]
            # text_encoding = torch.mean(text_encoding, dim=1,keepdim=True)
            # text_encoding = text_encoding.squeeze(1)
            # text_encoding = self.fc(text_encoding)
            # # print('models encoding:',text_encoding) #1,1,768
            # if i == 0:
            #     text_encoding_bert = text_encoding
            # else:
            #     text_encoding_bert = torch.cat([text_encoding_bert,text_encoding],dim=0)

            root_idx_p.append(idx_propagation_list.index(root_idx_item))
        root_idx_p = torch.tensor(root_idx_p)
        # print('---model root_idx_p---:',root_idx_p)


        for idx in data.x:
            idx = idx.item()
            text_embedding = self.embedding(torch.LongTensor(self.mid2regtextidx_dict[idx]).to(self.device))
            text_embedding = torch.unsqueeze(text_embedding,dim=1)
            text_encoding = self.gru(text_embedding,self.text_length)
            idx2encoding[idx] = text_encoding
            features_propagation.append(text_encoding.detach().cpu().numpy())


        # text_encoding_all = torch.tensor(text_encoding_bert)
        # print('models encoding_all:', text_encoding_bert.size())#8,128

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>comment>>>>>>>>>>>>>>>>>>>
        # for idx in
        dual_graph_mutual_index_p = []
        dual_graph_mutual_index_k = []
        for idx in data.x_knowledge_idx:
            idx = idx.item()
            if idx in self.token_dict:
                token_idx = torch.LongTensor([self.token_dict[idx]]).to(self.device)
                entity_embedding = self.embedding(token_idx)
                features_knowledge.append(entity_embedding.detach().cpu().numpy())
            elif idx in self.mid2regtextidx_dict:
                dual_graph_mutual_index_p.append(idx_propagation_list.index(idx))
                dual_graph_mutual_index_k.append(idx_knowledge_list.index(idx))
                features_knowledge.append(idx2encoding[idx].detach().cpu().numpy())
                    # features_knowledge.append(idx2encoding[idx].detach().cpu().numpy())
        # print('model dual_graph_mutual_index_p:',dual_graph_mutual_index_p)
        # print('model dual_graph_mutual_index_k:',dual_graph_mutual_index_k)
        features_propagation = torch.tensor(features_propagation)
        features_knowledge = torch.tensor(features_knowledge)

        # emb_propagation = self.embedding(vertices_propagation)
        if self.instance_norm:
            features_propagation = self.norm(features_propagation)

        # emb_knowledge = self.embedding(vertices_knowledge)
        if self.instance_norm:
            features_knowledge = self.norm(features_knowledge)

        x_propagation = features_propagation
        x_knowledge = features_knowledge
        # print('x_propagation:',features_propagation.size()) #7,1,128
        # print('x_knowleddge:',features_knowledge.size()) #771,1,128
        x_propagation = torch.squeeze(x_propagation,dim=1)
        x_knowledge = torch.squeeze(x_knowledge,dim=1)
        # print('x_propagation:', x_propagation.size())  # 7,1,128
        # print('x_knowleddge:', x_knowledge.size())  # 771,1,128
        # print('adj_propagation:',adjs_propagation[0].size())
        # print('adj_knowledge:',adjs_knowledge[0].size())

        x_propagation = x_propagation.to(self.device)
        x_knowledge = x_knowledge.to(self.device)
        x_propagation_node_indices = [data.edge_index,data.x_propagation_node_indices_1_edge_index,data.x_propagation_node_indices_2_edge_index]
        x_propagation_node_values = [data.edge_values,data.x_propagation_node_values_1,data.x_propagation_node_values_2]
        x_progation_edge_num = [data.edge_num,data.x_propagation_edge_num_1,data.x_propagation_edge_num_2]

        x_knowledge_node_indices = [data.x_knowledge_node_indices_0_edge_index, data.x_knowledge_node_indices_1_edge_index,
                                      data.x_knowledge_node_indices_2_edge_index]
        x_knowledge_node_values = [data.x_knowledge_node_values_0, data.x_knowledge_node_values_1,
                                     data.x_knowledge_node_values_2]
        x_knowledge_edge_num = [data.x_knowledge_edge_num_0, data.x_knowledge_edge_num_1,
                                data.x_knowledge_edge_num_2]
        # r = []
        # TODO: dynamic_graph那一篇paper 代码和文中所提不太一样
        last_x = x_propagation
        for i, gcn_layer in enumerate(self.layer_stack_propagation):

            x_gcn = gcn_layer(x_propagation, x_propagation_node_indices[i], x_propagation_node_values[i], root_idx_p, data.x_propagation_node_num, x_progation_edge_num[i],data.batch)
            # print('x_gcn:',x_gcn.size())
            x_propagation = self.bn_stack_propagation[i](x_gcn)
            x_knowledge = self.bn_stack_knowledge[i](self.layer_stack_knowledge[i](x_knowledge,x_knowledge_node_indices[i],x_knowledge_node_values[i],data.x_knowledge_node_num,x_knowledge_edge_num[i]))
            # if i < len(self.layer_stack) - 1:
            x_propagation = F.relu(x_propagation)
            x_knowledge = F.relu(x_knowledge)

            x_propagation = F.dropout(x_propagation, self.dropout, training=self.training)
            x_knowledge = F.dropout(x_knowledge,self.dropout,training = self.training)

            # print('attention test models:')
            # print('x_propagation size:',x_propagation.size())
            # print('x_knowledge size:',x_knowledge.size())
            x_propagation, x_knowledge = self.temporal_cells[i](x_propagation, x_knowledge,dual_graph_mutual_index_p,dual_graph_mutual_index_k,last_x,data.x_propagation_node_num,data.x_knowledge_node_num)  # temporal encoding

        x_propagation = F.relu(x_propagation)
        # x_knowledge = F.relu(x_knowledge)
        # x = torch.cat([x_propagation,x_knowledge],1)
        # print('---model x propagation---:', x_propagation.size())
        x = self.mean(x_propagation, data.batch)
        x_fusion = x.squeeze(1)
        # print('model x size:',x.size())
        #TODO: x,text_encoding normalized(?)
        # x_fusion = torch.cat([x,text_encoding_bert],dim=1)
        # print('---model x ---:', x.size()) #[32,128]
        x_fusion = F.relu(self.fc_rumor_1(x_fusion))
        x_fusion = self.fc_rumor_2(x_fusion)
        # x = self.mask(x, vertices)
        # print('----result before sigmoid----:',x)
        x_fusion = torch.sigmoid(x_fusion)
        # print('----result after sigmoid----:',x,x.size())
        return x_fusion
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#post节点的初始embedding换成gru,加最后bert的信息
class dual_DynamicGCN_nBatch_completed_gru(nn.Module):
    def __init__(self, token_dict, mid2regtextidx_dict, mid2bert_tokenizer, bert_path, n_output,config,device, n_hidden=3, dropout=0.2, instance_norm=False):
        super(dual_DynamicGCN_nBatch_completed_gru, self).__init__()
        self.token_dict = token_dict
        self.mid2regtextidx_dict = mid2regtextidx_dict
        self.mid2bert_tokenizer = mid2bert_tokenizer
        self.config = config
        self.dropout = dropout
        self.instance_norm = instance_norm
        if self.instance_norm:
            self.norm = nn.InstanceNorm1d(self.config.graph_embedding_dim, momentum=0.0, affine=True)
        self.n_output = n_output
        self.device = device
        self.text_length = torch.tensor([self.config.text_max_length]).to(self.device)

        #-----------------------text-----------------------------------
        self.embedding = nn.Embedding(len(self.token_dict), config.graph_embedding_dim)
        self.gru = GRUEncoder(self.config.graph_embedding_dim, self.config.hidden_dim)
        # self.embedding.weight = nn.Parameter(pretrained_emb)
        # self.embedding.weight.requires_grad = False

        self.n_feature = self.config.hidden_dim
        # self.n_input = pretrained_emb.size(0)

        #------------------ ----bert-----------------------------------
        modelConfig = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path, config=modelConfig)
        for param in self.bert.parameters():
            param.requires_grad = True
        bert_outputdim = self.bert.config.hidden_size
        self.fc = nn.Linear(bert_outputdim, 128)


        #------------------------temporal GCN--------------------------
        self.layer_stack_propagation = nn.ModuleList()  # TODO class initiate
        self.layer_stack_knowledge = nn.ModuleList()
        self.bn_stack_propagation = nn.ModuleList()
        self.bn_stack_knowledge = nn.ModuleList()
        self.temporal_cells = nn.ModuleList()
        for i in range(n_hidden):
            self.layer_stack_propagation.append(PropagationGraphConvLayer_nBatch_geometric(self.n_feature, self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_propagation.append(nn.BatchNorm1d(self.n_feature))
            self.layer_stack_knowledge.append(KnowledgeGraphConvLayer_nBatch_geometric(self.n_feature,self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_knowledge.append(nn.BatchNorm1d(self.n_feature))
            self.temporal_cells.append(TemporalEncoding_nBatch_geometric(self.n_feature, self.n_feature))


        self.mean = Mean_nBatch_geometric(self.device,self.n_feature,self.n_feature)
        # self.linear = nn.Linear(self.n_feature,self.n_output)
        self.fc_rumor_1 = nn.Linear(256,100)
        self.fc_rumor_2 = nn.Linear(100,2)

    def forward(self, data):
        idx2encoding = {}
        features_propagation = []
        features_knowledge = []
        # text_encoding_bert = []

        #将图节点tensor转换为list
        idx_propagation_list = data.x.squeeze(1).cpu().numpy().tolist()
        idx_knowledge_list = data.x_knowledge_idx.squeeze(1).cpu().numpy().tolist()
        # print('---model idx_propagation_list---:', idx_propagation_list)

        #将root_idx对应node2idx转换为在图中的
        root_idx_p = []
        for i,root_idx in enumerate(data.root_idx):
            root_idx_item = root_idx.item()
            # bert
            input_ids = self.mid2bert_tokenizer[root_idx_item]['input_ids'].to(self.device)
            attention_mask_bert = self.mid2bert_tokenizer[root_idx_item]['attention_mask'].to(self.device)
            text_encoding = self.bert(input_ids,attention_mask=attention_mask_bert)
            text_encoding = text_encoding[0]
            text_encoding = torch.mean(text_encoding, dim=1,keepdim=True)
            text_encoding = text_encoding.squeeze(1)
            text_encoding = self.fc(text_encoding)
            # print('models encoding:',text_encoding) #1,1,768
            if i == 0:
                text_encoding_bert = text_encoding
            else:
                text_encoding_bert = torch.cat([text_encoding_bert,text_encoding],dim=0)

            root_idx_p.append(idx_propagation_list.index(root_idx_item))
        root_idx_p = torch.tensor(root_idx_p)
        # print('---model root_idx_p---:',root_idx_p)


        for idx in data.x:
            idx = idx.item()
            text_embedding = self.embedding(torch.LongTensor(self.mid2regtextidx_dict[idx]).to(self.device))
            text_embedding = torch.unsqueeze(text_embedding,dim=1)
            text_encoding_gru = self.gru(text_embedding,self.text_length)
            idx2encoding[idx] = text_encoding_gru
            features_propagation.append(text_encoding_gru.detach().cpu().numpy())


        # text_encoding_all = torch.tensor(text_encoding_bert)
        # print('models encoding_all:', text_encoding_bert.size())#8,128

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>comment>>>>>>>>>>>>>>>>>>>
        # for idx in
        dual_graph_mutual_index_p = []
        dual_graph_mutual_index_k = []
        for idx in data.x_knowledge_idx:
            idx = idx.item()
            if idx in self.token_dict:
                token_idx = torch.LongTensor([self.token_dict[idx]]).to(self.device)
                entity_embedding = self.embedding(token_idx)
                features_knowledge.append(entity_embedding.detach().cpu().numpy())
            elif idx in self.mid2regtextidx_dict:
                dual_graph_mutual_index_p.append(idx_propagation_list.index(idx))
                dual_graph_mutual_index_k.append(idx_knowledge_list.index(idx))
                features_knowledge.append(idx2encoding[idx].detach().cpu().numpy())
                # features_knowledge.append(idx2encoding[idx].detach().cpu().numpy())
        # print('model dual_graph_mutual_index_p:',dual_graph_mutual_index_p)
        # print('model dual_graph_mutual_index_k:',dual_graph_mutual_index_k)
        features_propagation = torch.tensor(features_propagation)
        features_knowledge = torch.tensor(features_knowledge)

        # emb_propagation = self.embedding(vertices_propagation)
        if self.instance_norm:
            features_propagation = self.norm(features_propagation)

        # emb_knowledge = self.embedding(vertices_knowledge)
        if self.instance_norm:
            features_knowledge = self.norm(features_knowledge)

        x_propagation = features_propagation
        x_knowledge = features_knowledge
        # print('x_propagation:',features_propagation.size()) #7,1,128
        # print('x_knowleddge:',features_knowledge.size()) #771,1,128
        x_propagation = torch.squeeze(x_propagation,dim=1)
        x_knowledge = torch.squeeze(x_knowledge,dim=1)
        # print('x_propagation:', x_propagation.size())  # 7,1,128
        # print('x_knowleddge:', x_knowledge.size())  # 771,1,128
        # print('adj_propagation:',adjs_propagation[0].size())
        # print('adj_knowledge:',adjs_knowledge[0].size())

        x_propagation = x_propagation.to(self.device)
        x_knowledge = x_knowledge.to(self.device)
        x_propagation_node_indices = [data.edge_index,data.x_propagation_node_indices_1_edge_index,data.x_propagation_node_indices_2_edge_index]
        x_propagation_node_values = [data.edge_values,data.x_propagation_node_values_1,data.x_propagation_node_values_2]
        x_progation_edge_num = [data.edge_num,data.x_propagation_edge_num_1,data.x_propagation_edge_num_2]

        x_knowledge_node_indices = [data.x_knowledge_node_indices_0_edge_index, data.x_knowledge_node_indices_1_edge_index,
                                      data.x_knowledge_node_indices_2_edge_index]
        x_knowledge_node_values = [data.x_knowledge_node_values_0, data.x_knowledge_node_values_1,
                                     data.x_knowledge_node_values_2]
        x_knowledge_edge_num = [data.x_knowledge_edge_num_0, data.x_knowledge_edge_num_1,
                                data.x_knowledge_edge_num_2]
        # r = []
        # TODO: dynamic_graph那一篇paper 代码和文中所提不太一样
        last_x = x_propagation
        for i, gcn_layer in enumerate(self.layer_stack_propagation):

            x_gcn = gcn_layer(x_propagation, x_propagation_node_indices[i], x_propagation_node_values[i], root_idx_p, data.x_propagation_node_num, x_progation_edge_num[i],data.batch)
            # print('x_gcn:',x_gcn.size())
            x_propagation = self.bn_stack_propagation[i](x_gcn)
            x_knowledge = self.bn_stack_knowledge[i](self.layer_stack_knowledge[i](x_knowledge,x_knowledge_node_indices[i],x_knowledge_node_values[i],data.x_knowledge_node_num,x_knowledge_edge_num[i]))
            # if i < len(self.layer_stack) - 1:
            x_propagation = F.relu(x_propagation)
            x_knowledge = F.relu(x_knowledge)

            x_propagation = F.dropout(x_propagation, self.dropout, training=self.training)
            x_knowledge = F.dropout(x_knowledge,self.dropout,training = self.training)

            # print('attention test models:')
            # print('x_propagation size:',x_propagation.size())
            # print('x_knowledge size:',x_knowledge.size())
            x_propagation, x_knowledge = self.temporal_cells[i](x_propagation, x_knowledge,dual_graph_mutual_index_p,dual_graph_mutual_index_k,last_x,data.x_propagation_node_num,data.x_knowledge_node_num)  # temporal encoding

        x_propagation = F.relu(x_propagation)
        # x_knowledge = F.relu(x_knowledge)
        # x = torch.cat([x_propagation,x_knowledge],1)
        # print('---model x propagation---:', x_propagation.size())
        x = self.mean(x_propagation, data.batch)
        x = x.squeeze(1)
        # print('model x size:',x.size())
        #TODO: x,text_encoding normalized(?)
        x_fusion = torch.cat([x,text_encoding_bert],dim=1)
        # print('---model x ---:', x.size()) #[32,128]
        x_fusion = F.relu(self.fc_rumor_1(x_fusion))
        x_fusion = self.fc_rumor_2(x_fusion)
        # x = self.mask(x, vertices)
        # print('----result before sigmoid----:',x)
        x_fusion = torch.sigmoid(x_fusion)
        # print('----result after sigmoid----:',x,x.size())
        return x_fusion
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#只用gru， 文本求个mean
class dual_DynamicGCN_nBatch_gru(nn.Module):
    def __init__(self, token_dict, mid2regtextidx_dict, bert_path, n_output,config,device, n_hidden=3, dropout=0.2, instance_norm=False):
        super(dual_DynamicGCN_nBatch_gru, self).__init__()
        self.token_dict = token_dict
        self.mid2regtextidx_dict = mid2regtextidx_dict
        self.config = config
        self.dropout = dropout
        self.instance_norm = instance_norm
        if self.instance_norm:
            self.norm = nn.InstanceNorm1d(self.config.graph_embedding_dim, momentum=0.0, affine=True)
        self.n_output = n_output
        self.device = device
        self.text_length = torch.tensor([self.config.text_max_length]).to(self.device)

        #-----------------------text-----------------------------------
        self.embedding = nn.Embedding(len(self.token_dict), config.graph_embedding_dim)
        self.gru = BiLSTMEncoder(self.config.graph_embedding_dim, self.config.hidden_dim)
        # self.embedding.weight = nn.Parameter(pretrained_emb)
        # self.embedding.weight.requires_grad = False

        self.n_feature = self.config.hidden_dim
        # self.n_input = pretrained_emb.size(0)

        #------------------ ----bert-----------------------------------
        # modelConfig = BertConfig.from_pretrained(bert_path)
        # self.bert = BertModel.from_pretrained(bert_path, config=modelConfig)
        # for param in self.bert.parameters():
        #     param.requires_grad = True
        # bert_outputdim = self.bert.config.hidden_size
        # self.fc = nn.Linear(bert_outputdim, 128)


        #------------------------temporal GCN--------------------------
        self.layer_stack_propagation = nn.ModuleList()  # TODO class initiate
        self.layer_stack_knowledge = nn.ModuleList()
        self.bn_stack_propagation = nn.ModuleList()
        self.bn_stack_knowledge = nn.ModuleList()
        self.temporal_cells = nn.ModuleList()
        for i in range(n_hidden):
            self.layer_stack_propagation.append(PropagationGraphConvLayer_nBatch_geometric(self.n_feature, self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_propagation.append(nn.BatchNorm1d(self.n_feature))
            self.layer_stack_knowledge.append(KnowledgeGraphConvLayer_nBatch_geometric(self.n_feature,self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_knowledge.append(nn.BatchNorm1d(self.n_feature))
            self.temporal_cells.append(TemporalEncoding_nBatch_geometric(self.n_feature, self.n_feature))


        self.mean = Mean_nBatch_geometric(self.device,self.n_feature,self.n_feature)
        # self.linear = nn.Linear(self.n_feature,self.n_output)
        self.fc_rumor_1 = nn.Linear(256,100)
        self.fc_rumor_2 = nn.Linear(100,2)

    def forward(self, data):
        idx2encoding = {}
        features_propagation = []
        features_knowledge = []
        # text_encoding_bert = []

        #将图节点tensor转换为list
        idx_propagation_list = data.x.squeeze(1).cpu().numpy().tolist()
        idx_knowledge_list = data.x_knowledge_idx.squeeze(1).cpu().numpy().tolist()
        # print('---model idx_propagation_list---:', idx_propagation_list)

        #将root_idx对应node2idx转换为在图中的
        root_idx_p = []
        for i,root_idx in enumerate(data.root_idx):
            root_idx_item = root_idx.item()
            # bert
            # input_ids = self.mid2regtextidx_dict[root_idx_item]['input_ids'].to(self.device)
            # attention_mask_bert = self.mid2regtextidx_dict[root_idx_item]['attention_mask'].to(self.device)
            # text_encoding = self.bert(input_ids,attention_mask=attention_mask_bert)
            # text_encoding = text_encoding[0]
            # text_encoding = torch.mean(text_encoding, dim=1,keepdim=True)
            # text_encoding = text_encoding.squeeze(1)
            # text_encoding = self.fc(text_encoding)
            # # print('models encoding:',text_encoding) #1,1,768
            # if i == 0:
            #     text_encoding_bert = text_encoding
            # else:
            #     text_encoding_bert = torch.cat([text_encoding_bert,text_encoding],dim=0)

            root_idx_p.append(idx_propagation_list.index(root_idx_item))
        root_idx_p = torch.tensor(root_idx_p)
        # print('---model root_idx_p---:',root_idx_p)


        for idx in data.x:
            idx = idx.item()
            text_embedding = self.embedding(torch.LongTensor(self.mid2regtextidx_dict[idx]).to(self.device))
            text_embedding = torch.unsqueeze(text_embedding,dim=1)
            text_encoding = self.gru(text_embedding,self.text_length)
            # print('models gru text_encoding:',text_encoding.size())
            idx2encoding[idx] = text_encoding
            features_propagation.append(text_encoding.detach().cpu().numpy())


        # text_encoding_all = torch.tensor(text_encoding_bert)
        # print('models encoding_all:', text_encoding_bert.size())#8,128

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>comment>>>>>>>>>>>>>>>>>>>
        # for idx in
        # dual_graph_mutual_index_p = []
        # dual_graph_mutual_index_k = []
        # for idx in data.x_knowledge_idx:
        #     idx = idx.item()
        #     if idx in self.token_dict:
        #         token_idx = torch.LongTensor([self.token_dict[idx]]).to(self.device)
        #         entity_embedding = self.embedding(token_idx)
        #         features_knowledge.append(entity_embedding.detach().cpu().numpy())
        #     elif idx in self.mid2regtextidx_dict:
        #         dual_graph_mutual_index_p.append(idx_propagation_list.index(idx))
        #         dual_graph_mutual_index_k.append(idx_knowledge_list.index(idx))
        #         features_knowledge.append(idx2encoding[idx].detach().cpu().numpy())
        #             # features_knowledge.append(idx2encoding[idx].detach().cpu().numpy())
        # # print('model dual_graph_mutual_index_p:',dual_graph_mutual_index_p)
        # # print('model dual_graph_mutual_index_k:',dual_graph_mutual_index_k)
        features_propagation = torch.tensor(features_propagation)
        # features_knowledge = torch.tensor(features_knowledge)

        # emb_propagation = self.embedding(vertices_propagation)
        if self.instance_norm:
            features_propagation = self.norm(features_propagation)

        # emb_knowledge = self.embedding(vertices_knowledge)
        if self.instance_norm:
            features_knowledge = self.norm(features_knowledge)

        x_propagation = features_propagation
        # x_knowledge = features_knowledge
        # print('x_propagation:',features_propagation.size()) #7,1,128
        # print('x_knowleddge:',features_knowledge.size()) #771,1,128
        # x_propagation = torch.squeeze(x_propagation,dim=1)
        # x_knowledge = torch.squeeze(x_knowledge,dim=1)
        # print('x_propagation:', x_propagation.size())  # 7,1,128
        # print('x_knowleddge:', x_knowledge.size())  # 771,1,128
        # print('adj_propagation:',adjs_propagation[0].size())
        # print('adj_knowledge:',adjs_knowledge[0].size())

        x_propagation = x_propagation.to(self.device)
        # x_knowledge = x_knowledge.to(self.device)
        # x_propagation_node_indices = [data.edge_index,data.x_propagation_node_indices_1_edge_index,data.x_propagation_node_indices_2_edge_index]
        # x_propagation_node_values = [data.edge_values,data.x_propagation_node_values_1,data.x_propagation_node_values_2]
        # x_progation_edge_num = [data.edge_num,data.x_propagation_edge_num_1,data.x_propagation_edge_num_2]
        #
        # x_knowledge_node_indices = [data.x_knowledge_node_indices_0_edge_index, data.x_knowledge_node_indices_1_edge_index,
        #                               data.x_knowledge_node_indices_2_edge_index]
        # x_knowledge_node_values = [data.x_knowledge_node_values_0, data.x_knowledge_node_values_1,
        #                              data.x_knowledge_node_values_2]
        # x_knowledge_edge_num = [data.x_knowledge_edge_num_0, data.x_knowledge_edge_num_1,
        #                         data.x_knowledge_edge_num_2]
        # # r = []
        # # TODO: dynamic_graph那一篇paper 代码和文中所提不太一样
        # last_x = x_propagation
        # for i, gcn_layer in enumerate(self.layer_stack_propagation):
        #
        #     x_gcn = gcn_layer(x_propagation, x_propagation_node_indices[i], x_propagation_node_values[i], root_idx_p, data.x_propagation_node_num, x_progation_edge_num[i],data.batch)
        #     # print('x_gcn:',x_gcn.size())
        #     x_propagation = self.bn_stack_propagation[i](x_gcn)
        #     x_knowledge = self.bn_stack_knowledge[i](self.layer_stack_knowledge[i](x_knowledge,x_knowledge_node_indices[i],x_knowledge_node_values[i],data.x_knowledge_node_num,x_knowledge_edge_num[i]))
        #     # if i < len(self.layer_stack) - 1:
        #     x_propagation = F.relu(x_propagation)
        #     x_knowledge = F.relu(x_knowledge)
        #
        #     x_propagation = F.dropout(x_propagation, self.dropout, training=self.training)
        #     x_knowledge = F.dropout(x_knowledge,self.dropout,training = self.training)
        #
        #     # print('attention test models:')
        #     # print('x_propagation size:',x_propagation.size())
        #     # print('x_knowledge size:',x_knowledge.size())
        #     x_propagation, x_knowledge = self.temporal_cells[i](x_propagation, x_knowledge,dual_graph_mutual_index_p,dual_graph_mutual_index_k,last_x,data.x_propagation_node_num,data.x_knowledge_node_num)  # temporal encoding
        #
        # x_propagation = F.relu(x_propagation)
        # x_knowledge = F.relu(x_knowledge)
        # x = torch.cat([x_propagation,x_knowledge],1)
        # print('---model x propagation---:', x_propagation.size())
        # print('models gru x_propagation:',x_propagation)
        # print('models gru x_propagation.size():', x_propagation.size())
        x_fusion = self.mean(x_propagation, data.batch)
        # print(x.size()) #32,256
        # x_fusion = x.squeeze(1)
        # print('models gru x_fusion:',x_fusion)
        # print('model x size:',x.size())
        #TODO: x,text_encoding normalized(?)
        # x_fusion = torch.cat([x,text_encoding_bert],dim=1)
        # print('---model x ---:', x.size()) #[32,128]
        x_fusion = F.leaky_relu(self.fc_rumor_1(x_fusion))
        x_fusion = self.fc_rumor_2(x_fusion)
        # x = self.mask(x, vertices)
        # print('----result before sigmoid----:',x)
        x_fusion = torch.sigmoid(x_fusion)
        # print('----result after sigmoid----:',x,x.size())
        return x_fusion
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#只有bert的基础模型
class dual_DynamicGCN_nBatch_BERT(nn.Module):
    def __init__(self, token_dict, mid2bert_tokenizer, bert_path, n_output,config,device, n_hidden=3, dropout=0.2, instance_norm=False):
        super(dual_DynamicGCN_nBatch_BERT, self).__init__()
        self.token_dict = token_dict
        self.mid2bert_tokenizer = mid2bert_tokenizer
        self.config = config
        self.dropout = dropout
        self.instance_norm = instance_norm
        if self.instance_norm:
            self.norm = nn.InstanceNorm1d(self.config.graph_embedding_dim, momentum=0.0, affine=True)
        self.n_output = n_output
        self.device = device
        self.text_length = torch.tensor([self.config.text_max_length]).to(self.device)

        #-----------------------text-----------------------------------
        self.embedding = nn.Embedding(len(self.token_dict), config.graph_embedding_dim)
        self.gru = GRUEncoder(self.config.graph_embedding_dim, self.config.hidden_dim)
        # self.embedding.weight = nn.Parameter(pretrained_emb)
        # self.embedding.weight.requires_grad = False

        self.n_feature = self.config.hidden_dim
        # self.n_input = pretrained_emb.size(0)

        #------------------ ----bert-----------------------------------
        modelConfig = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path, config=modelConfig)
        for param in self.bert.parameters():
            param.requires_grad = True
        bert_outputdim = self.bert.config.hidden_size
        self.fc = nn.Linear(bert_outputdim, 128)


        #------------------------temporal GCN--------------------------
        self.layer_stack_propagation = nn.ModuleList()  # TODO class initiate
        self.layer_stack_knowledge = nn.ModuleList()
        self.bn_stack_propagation = nn.ModuleList()
        self.bn_stack_knowledge = nn.ModuleList()
        self.temporal_cells = nn.ModuleList()
        for i in range(n_hidden):
            self.layer_stack_propagation.append(PropagationGraphConvLayer_nBatch_geometric(self.n_feature, self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_propagation.append(nn.BatchNorm1d(self.n_feature))
            self.layer_stack_knowledge.append(KnowledgeGraphConvLayer_nBatch_geometric(self.n_feature,self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_knowledge.append(nn.BatchNorm1d(self.n_feature))
            self.temporal_cells.append(TemporalEncoding_nBatch_geometric(self.n_feature, self.n_feature))


        self.mean = Mean_nBatch_geometric(self.device,self.n_feature,self.n_output)
        # self.linear = nn.Linear(self.n_feature,self.n_output)
        self.fc_rumor_1 = nn.Linear(128,100)
        self.fc_rumor_2 = nn.Linear(100,2)

    def forward(self, data):
        idx2encoding = {}
        features_propagation = []
        features_knowledge = []
        # text_encoding_bert = []

        #将图节点tensor转换为list
        idx_propagation_list = data.x.squeeze(1).cpu().numpy().tolist()
        idx_knowledge_list = data.x_knowledge_idx.squeeze(1).cpu().numpy().tolist()
        # print('---model idx_propagation_list---:', idx_propagation_list)

        #将root_idx对应node2idx转换为在图中的
        root_idx_p = []
        for i,root_idx in enumerate(data.root_idx):
            root_idx_item = root_idx.item()
            # bert
            input_ids = self.mid2bert_tokenizer[root_idx_item]['input_ids'].to(self.device)
            attention_mask_bert = self.mid2bert_tokenizer[root_idx_item]['attention_mask'].to(self.device)
            # print('model input ids:',input_ids)
            # print('model attention mask bert:',attention_mask_bert)
            text_encoding = self.bert(input_ids,attention_mask=attention_mask_bert)
            text_encoding = text_encoding[0]
            text_encoding = torch.mean(text_encoding, dim=1,keepdim=True)
            text_encoding = text_encoding.squeeze(1)
            text_encoding = F.relu(self.fc(text_encoding))
            # print('models encoding:',text_encoding) #1,1,768
            if i == 0:
                text_encoding_bert = text_encoding
            else:
                text_encoding_bert = torch.cat([text_encoding_bert,text_encoding],dim=0)

            root_idx_p.append(idx_propagation_list.index(root_idx_item))
        root_idx_p = torch.tensor(root_idx_p)
        # print('---model root_idx_p---:',root_idx_p)



        # x_propagation = F.relu(x_propagation)
        # x_knowledge = F.relu(x_knowledge)
        # x = torch.cat([x_propagation,x_knowledge],1)
        # print('---model x propagation---:', x_propagation.size())
        # x = self.mean(x_propagation, data.batch)
        # x = x.squeeze(1)
        #TODO: x,text_encoding normalized(?)
        # x_fusion = torch.cat([x,text_encoding_bert],dim=1)
        # print('---model x ---:', x.size()) #[32,128]
        x_fusion = F.relu(self.fc_rumor_1(text_encoding_bert))
        x_fusion = self.fc_rumor_2(x_fusion)
        # x = self.mask(x, vertices)
        # print('----result before sigmoid----:',x)
        x_fusion = torch.sigmoid(x_fusion)
        # print('----result after sigmoid----:',x,x.size())
        return x_fusion
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#=================================================

class dual_DynamicGCN(nn.Module):
    def __init__(self, token_dict,mid2regtextidx_dict,n_output,config,device, n_hidden=3, dropout=0.2, instance_norm=False):
        super(dual_DynamicGCN, self).__init__()
        self.token_dict = token_dict
        self.mid2regtextidx_dict = mid2regtextidx_dict
        self.config = config
        self.dropout = dropout
        self.instance_norm = instance_norm
        if self.instance_norm:
            self.norm = nn.InstanceNorm1d(self.config.graph_embedding_dim, momentum=0.0, affine=True)
        self.n_output = n_output
        self.device = device

        #text
        self.embedding = nn.Embedding(len(self.token_dict), config.graph_embedding_dim)
        self.gru = GRUEncoder(self.config.graph_embedding_dim, self.config.hidden_dim)
        # self.embedding.weight = nn.Parameter(pretrained_emb)
        # self.embedding.weight.requires_grad = False

        self.n_feature = self.config.hidden_dim
        # self.n_input = pretrained_emb.size(0)

        self.layer_stack_propagation = nn.ModuleList()  # TODO class initiate
        self.layer_stack_knowledge = nn.ModuleList()
        self.bn_stack_propagation = nn.ModuleList()
        self.bn_stack_knowledge = nn.ModuleList()
        self.temporal_cells = nn.ModuleList()
        for i in range(n_hidden):
            self.layer_stack_propagation.append(PropagationGraphConvLayer(self.n_feature, self.n_feature, self.n_feature,self.dropout,self.device))
            self.bn_stack_propagation.append(nn.BatchNorm1d(self.n_feature))
            self.layer_stack_knowledge.append(KnowledgeGraphConvLayer(self.n_feature, self.n_feature, self.n_feature,self.dropout))
            self.bn_stack_knowledge.append(nn.BatchNorm1d(self.n_feature))
            self.temporal_cells.append(TemporalEncoding(self.n_feature, self.n_feature))

        # self.layer_stack_propagation.append(SpGraphConvLayer(self.n_feature, self.n_feature))
        # self.layer_stack_knowledge.append(SpGraphConvLayer(self.n_feature,self.n_feature))
        # self.bn_stack.append(nn.BatchNorm1d(2*self.n_feature))
        self.linear = nn.Linear(self.n_feature,self.n_output)
        self.mask = MaskLinear()
        self.save_x = None
        self.text_length = torch.LongTensor(self.config.text_max_length).to(self.device)

    def forward(self, adj_propagation,adj_knowledge,idx_propagation,idx_knowledge,root_idx_p):
        idx2encoding = {}
        features_propagation = []
        features_knowledge = []
        idx_propagation = idx_propagation.squeeze(0)
        idx_knowledge = idx_knowledge.squeeze(0)
        # print('idx_propagation model:',idx_propagation.size())
        # print('idx_knowledge model:', idx_knowledge.size())
        # print('adj_propagation model:',adj_propagation[0].size())
        # print('adj_knowledge model:',adj_knowledge[0].size())
        # print('root_idx_p model:',root_idx_p)
        idx_propagation_list = idx_propagation.cpu().numpy().tolist()
        # print('idx_propagation_list:',len(idx_propagation_list))
        idx_knowledge_list = idx_knowledge.cpu().numpy().tolist()

        root_idx_p_new = []
        # for root_idx in root_idx_p:
        root_idx_item = root_idx_p.item()
        root_idx_p_new.append(idx_propagation_list.index(root_idx_item))
        root_idx_p_new = torch.LongTensor(root_idx_p_new).to(self.device)

        for idx in idx_propagation:
            idx = idx.item()
            text_embedding = self.embedding(torch.LongTensor(self.mid2regtextidx_dict[idx]).to(self.device))
            text_embedding = torch.unsqueeze(text_embedding,dim=1)
            # print('---text embedding---:',text_embedding.size()) #120,1,128

            text_encoding = self.gru(text_embedding,self.text_length) #120,1,256
            # print('text encoding:',text_encoding.size())
            idx2encoding[idx] = text_encoding
            features_propagation.append(text_encoding.detach().cpu().numpy())
        # for idx in
        dual_graph_mutual_index_p = []
        dual_graph_mutual_index_k = []
        for idx in idx_knowledge:
            idx = idx.item()
            if idx in self.token_dict:
                # print('idx:',idx,'token dict idx:',self.token_dict[idx],'token dict type:',type(self.token_dict[idx]))
                token_idx = torch.LongTensor([self.token_dict[idx]]).to(self.device)
                # print('token idx',token_idx)
                entity_embedding = self.embedding(token_idx)
                # print('entity embedding:',entity_embedding.size()) #1,128
                # print('embedding',entity_embedding)
                # except RuntimeError:
                # print('error idx:',idx)
                # else:
                features_knowledge.append(entity_embedding.detach().cpu().numpy())
            elif idx in self.mid2regtextidx_dict:
                dual_graph_mutual_index_p.append(idx_propagation_list.index(idx))
                dual_graph_mutual_index_k.append(idx_knowledge_list.index(idx))
                features_knowledge.append(idx2encoding[idx].detach().cpu().numpy())

        features_propagation = torch.tensor(features_propagation)
        features_knowledge = torch.tensor(features_knowledge)

        # emb_propagation = self.embedding(vertices_propagation)
        if self.instance_norm:
            features_propagation = self.norm(features_propagation)

        # emb_knowledge = self.embedding(vertices_knowledge)
        if self.instance_norm:
            features_knowledge = self.norm(features_knowledge)

        x_propagation = features_propagation
        x_knowledge = features_knowledge
        # print('x_propagation:',features_propagation.size()) #7,1,128
        # print('x_knowleddge:',features_knowledge.size()) #771,1,128
        x_propagation = torch.squeeze(x_propagation,dim=1)
        x_knowledge = torch.squeeze(x_knowledge,dim=1)
        # print('x_propagation:', x_propagation.size())  # 7,1,128
        # print('x_knowleddge:', x_knowledge.size())  # 771,1,128
        # print('adj_propagation:',adjs_propagation[0].size())
        # print('adj_knowledge:',adjs_knowledge[0].size())

        x_propagation = x_propagation.to(self.device)
        x_knowledge = x_knowledge.to(self.device)
        # r = []
        #TODO: dynamic_graph那一篇paper 代码和文中所提不太一样
        last_x = x_propagation
        for i, gcn_layer in enumerate(self.layer_stack_propagation):
            # last_x = x
            x_gcn = gcn_layer(x_propagation,adj_propagation[i],root_idx_p_new)
            # print('x_gcn:',x_gcn.size())
            x_propagation = self.bn_stack_propagation[i](x_gcn)
            x_knowledge = self.bn_stack_knowledge[i](self.layer_stack_knowledge[i](x_knowledge,adj_knowledge[i]))
            # if i < len(self.layer_stack) - 1:
            x_propagation = F.relu(x_propagation)
            x_knowledge = F.relu(x_knowledge)

            x_propagation = F.dropout(x_propagation, self.dropout, training=self.training)
            x_knowledge = F.dropout(x_knowledge,self.dropout,training = self.training)

            x_propagation, x_knowledge = self.temporal_cells[i](x_propagation, x_knowledge,\
                                                                dual_graph_mutual_index_p,\
                                                                dual_graph_mutual_index_k,last_x)  # temporal encoding

        # x_propagation = F.relu(x_propagation)
        # x_knowledge = F.relu(x_knowledge)
        # x = torch.cat([x_propagation,x_knowledge],1)
        x = torch.mean(x_propagation,dim=0)
        x = F.leaky_relu(self.linear(x))
        # x = self.mask(x, vertices)
        # print('----result before sigmoid----:',x)
        x = torch.sigmoid(x)
        # print('----result after sigmoid----:',x)
        return x

