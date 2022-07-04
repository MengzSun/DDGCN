import os
import sys
import time

# from path import *
from path_zh import *
from data import *

class dual_dynamic_graph_Config():

    def __init__(self):

        # basic
        self.model_name = "dual_dynamic_graph"
        self.graph_embedding_dim = 128
        self.hidden_dim = 128
        self.n_class = 1
        # self.batch_size = 16
        # self.epoch_num = 5
        self.report_step_num = 10
        self.dropout_rate = 0.5
        # self.learning_rate = 5e-5
        self.min_learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.patience = 2
        self.train = 0.7
        self.val = 0.1
        self.test = 0.2

        # task specific
        self.text_max_length = 30#120
        self.pad_idx = 6691
        self.basis_num = 2
        self.use_text = True
        self.k_hop = 1

        # train
        self.gpu_id = "0"

        # vocab
        # self.entity_concept_path = path_entity_concept_2id
        # self.relation_path = path_relation2id
        # self.token_path = path_token2id

        # init
        self.init()


    def init(self):
        ''' additional configuration '''

        # vocab
        # self.entity2id,   self.id2entity,   self.entity_size   = load_str_dict(self.entity_path, reverse=True)
        # self.relation2id, self.id2relation, self.relation_size = load_str_dict(self.relation_path, reverse=True)
        # self.token2id,    self.id2token,    self.token_size    = load_str_dict(self.token_path, reverse=True)
        self.entity_concept_size = 96787
        # self.relation_size = 10
        self.token_size = 258466

        # extra adjacent matrix number
        self.add_adj_size = 1  # selfloop

# class dual_dynamic_graph_Config_BERT():
#
#     def __init__(self):
#
#         # basic
#         self.model_name = "dual_dynamic_graph"
#         self.graph_embedding_dim = 128
#         self.hidden_dim = 128
#         self.n_class = 1
#         self.batch_size = 64
#         self.epoch_num = 200
#         self.report_step_num = 10
#         self.dropout_rate = 0.5
#         self.learning_rate = 1e-2
#         self.min_learning_rate = 1e-4
#         self.weight_decay = 1e-4
#         self.patience = 2
#         self.train = 0.8
#         self.val = 0.1
#         self.test = 0.1
#
#         # task specific
#         self.text_max_length = 30#120
#         self.pad_idx = 6691
#         self.basis_num = 2
#         self.use_text = True
#         self.k_hop = 1
#
#         # train
#         self.gpu_id = "0"
#
#         # vocab
#         # self.entity_concept_path = path_entity_concept_2id
#         # self.relation_path = path_relation2id
#         # self.token_path = path_token2id
#
#         # init
#         self.init()
#
#
#     def init(self):
#         ''' additional configuration '''
#
#         # vocab
#         # self.entity2id,   self.id2entity,   self.entity_size   = load_str_dict(self.entity_path, reverse=True)
#         # self.relation2id, self.id2relation, self.relation_size = load_str_dict(self.relation_path, reverse=True)
#         # self.token2id,    self.id2token,    self.token_size    = load_str_dict(self.token_path, reverse=True)
#         self.entity_concept_size = 96787
#         # self.relation_size = 10
#         self.token_size = 258466
#
#         # extra adjacent matrix number
#         self.add_adj_size = 1  # selfloop

class dual_dynamic_graph_Config_zh():

    def __init__(self):

        # basic
        self.model_name = "dual_dynamic_graph"
        self.graph_embedding_dim = 128
        self.hidden_dim = 128
        self.n_class = 1
        # self.batch_size = 16
        # self.epoch_num = 100#5
        # self.learning_rate = 5e-5
        self.report_step_num = 10
        self.dropout_rate = 0.5
        self.min_learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.patience = 2
        self.train = 0.7
        self.val = 0.1
        self.test = 0.2

        # task specific
        self.text_max_length = 30#120
        self.pad_idx = 6691
        self.basis_num = 2
        self.use_text = True
        self.k_hop = 1

        # train
        self.gpu_id = "0"

        # vocab
        # self.entity_concept_path = path_entity_concept_2id
        # self.relation_path = path_relation2id
        # self.token_path = path_token2id

        # init
        self.init()


    def init(self):
        ''' additional configuration '''

        # vocab
        # self.entity2id,   self.id2entity,   self.entity_size   = load_str_dict(self.entity_path, reverse=True)
        # self.relation2id, self.id2relation, self.relation_size = load_str_dict(self.relation_path, reverse=True)
        # self.token2id,    self.id2token,    self.token_size    = load_str_dict(self.token_path, reverse=True)
        self.entity_concept_size = 96787
        # self.relation_size = 10
        self.token_size = 258466

        # extra adjacent matrix number
        self.add_adj_size = 1  # selfloop

# class dual_dynamic_graph_Config_zh_BERT():
#
#     def __init__(self):
#
#         # basic
#         self.model_name = "dual_dynamic_graph"
#         self.graph_embedding_dim = 128
#         self.hidden_dim = 128
#         self.n_class = 1
#         self.batch_size = 16
#         self.epoch_num = 200
#         self.report_step_num = 10
#         self.dropout_rate = 0.5
#         self.learning_rate = 1e-3
#         self.min_learning_rate = 1e-4
#         self.weight_decay = 1e-4
#         self.patience = 2
#         self.train = 0.8
#         self.val = 0.1
#         self.test = 0.1
#
#         # task specific
#         self.text_max_length = 30#120
#         self.pad_idx = 6691
#         self.basis_num = 2
#         self.use_text = True
#         self.k_hop = 1
#
#         # train
#         self.gpu_id = "0"
#
#         # vocab
#         # self.entity_concept_path = path_entity_concept_2id
#         # self.relation_path = path_relation2id
#         # self.token_path = path_token2id
#
#         # init
#         self.init()
#
#
#     def init(self):
#         ''' additional configuration '''
#
#         # vocab
#         # self.entity2id,   self.id2entity,   self.entity_size   = load_str_dict(self.entity_path, reverse=True)
#         # self.relation2id, self.id2relation, self.relation_size = load_str_dict(self.relation_path, reverse=True)
#         # self.token2id,    self.id2token,    self.token_size    = load_str_dict(self.token_path, reverse=True)
#         self.entity_concept_size = 96787
#         # self.relation_size = 10
#         self.token_size = 258466
#
#         # extra adjacent matrix number
#         self.add_adj_size = 1  # selfloop



