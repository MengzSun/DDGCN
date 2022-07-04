#处理weibo数据集，将其最终处理为dynamic_gcn的形式
#结构信息、文本信息、知识信息
#时间分割 time_split设为args参数，default=3
#知识图谱 probase & YAGO

#有一个问题 如果采isA关系，那么post和实体节点之间也要用tf-idf关系
#如果用三元组（知识图谱中的三元组关系），post和节点之间用has关系，但有一个问题三元组中关系太多了，应该也没啥用！
#还是采用第一个做法
import os
import pandas as pd
import datetime as dt
from dateutil import parser
from dateutil import rrule
import re
import numpy as np
from math import log
import itertools
from collections import defaultdict
from tqdm import tqdm
from scipy import sparse
import csv
import jieba

pheme_clean_path = '../data/weibo/weibo_clean/'
pheme_senti_path = '../data/weibo/weibo_senti/'
pheme_entity_path = '../data/weibo/weibo_entity/'
pheme_concept_yago_path = '../data/weibo/weibo_concept/'

pheme_temporal_path = '../data/weibo/weibo_temporal_data/'

#统计所有回复数量，取最小 最小为 1！！！
#时间转换
def stat_num():
    pheme_files = os.listdir(pheme_clean_path)
    min_num = 100
    for i in range(len(pheme_files)):
        mid_lst = []
        file = pheme_files[i]
        file_df = pd.read_csv(pheme_clean_path+file)
        for j in range(len(file_df['mid'])):
            mid = file_df['mid'][j]
            mid_lst.append(mid)
        num = len(list(set(mid_lst)))
        if num < min_num:
            min_num = num
    print(min_num)

#---------------数据清洗、转化--------------------
def month_trans(mon):
    mon_dic = {}
    mon_dic['Jan'] = 1
    mon_dic['Feb'] = 2
    mon_dic['Mar'] = 3
    mon_dic['Apr'] = 4
    mon_dic['May'] = 5
    mon_dic['Jun'] = 6
    mon_dic['Jul'] = 7
    mon_dic['Aug'] = 8
    mon_dic['Sep'] = 9
    mon_dic['Oct'] = 10
    mon_dic['Nov'] = 11
    mon_dic['Dec'] = 12
    return mon_dic[mon]

def trans_time(t, t_init):
    t = t.split(' ')
    t_exct = t[3].split(':')
    t_init = t_init.split(' ')
    t_init_exct = t_init[3].split(':')
    date_1 = dt.datetime(int(t[5]),month_trans(t[1]),int(t[2]),int(t_exct[0]),int(t_exct[1]),int(t_exct[2]))
    date_0 = dt.datetime(int(t_init[5]), month_trans(t_init[1]), int(t_init[2]), int(t_init_exct[0]), int(t_init_exct[1]), int(t_init_exct[2]))
    interval = (date_1 - date_0).seconds
    return interval

def trans_time_zh(t, t_init):
    # t = t.split(' ')
    # t_exct = t[3].split(':')
    # t_init = t_init.split(' ')
    # t_init_exct = t_init[3].split(':')
    # date_1 = dt.datetime(int(t[5]),month_trans(t[1]),int(t[2]),int(t_exct[0]),int(t_exct[1]),int(t_exct[2]))
    # date_0 = dt.datetime(int(t_init[5]), month_trans(t_init[1]), int(t_init[2]), int(t_init_exct[0]), int(t_init_exct[1]), int(t_init_exct[2]))
    # interval = (date_1 - date_0).seconds
    interval = t-t_init
    return interval

def senti_trans(senti):
    senti_dic = {}
    senti_dic['neg'] = -1
    senti_dic['neu'] = 0
    senti_dic['pos'] = 1
    return senti_dic[senti]
# trans_time()
def clean_text_en(text):
    r1 = "\n"
    r2 = '\r'
    r3 = '\t'
    text = re.sub(r1,' ',text)
    text = re.sub(r2,' ',text)
    text = re.sub(r3,' ',text)
    # text = re.sub(r1,'',text)
    return text

def read_data():
    tree_dic = {}
    #parent\text\t\entity\concept\senti
    #-------------------------------------------------
    pheme_files_entity = os.listdir(pheme_entity_path)
    for i in range(len(pheme_files_entity)):
        file = pheme_files_entity[i].split('.')[0]
        # print('file:',file)
        tree_dic[file] = {}
        file_df_entity = pd.read_csv(pheme_entity_path + file + '.csv')
        # t_init =
        # init_dic = {}
        # for j in range(len(file_df_entity['mid'])):
        #     mid = file_df_entity['mid'][j]
        #     t = file_df_entity['t'][j]
        #     init_dic[mid] = t
        # t_init = init_dic[int(file)]
        # t_init = 0
        j_init = 0
        for j in range(len(file_df_entity['mid'])):
            # mid = file_df_entity['mid'][j]
            # if not mid in tree_dic[file]:
            if file_df_entity['parent'][j] == 'None':
                # t_init = file_df_entity['t'][j]
                j_init = j
        t_init = file_df_entity['t'][j_init]
        # print(t_init)


        for j in range(len(file_df_entity['mid'])):
            mid = file_df_entity['mid'][j]
            if not mid in tree_dic[file]:
                # if file_df_entity['parent'][j] == 'None':
                #     t_init = file_df_entity['t'][j]
                tree_dic[file][mid] = {}
                tree_dic[file][mid]['parent'] = file_df_entity['parent'][j].strip('\t')
                # tree_dic[file][mid]['text'] = ('').join((' ').join(file_df_entity['text'][j].split('\n')).split('\n'))
                tree_dic[file][mid]['text'] = clean_text_en(file_df_entity['text'][j])
                t_trans = trans_time_zh(file_df_entity['t'][j], t_init)
                tree_dic[file][mid]['t'] = t_trans
                # print("read data t:",t_trans)
                entity = [ent.strip(' ').strip('\'').strip(' ') for ent in file_df_entity['entity'][j].strip('\t').strip('[').strip(']').split(',')]
                # if '[\'\']' in entity:
                #     entity.remove('[\'\']')
                for ent in entity:
                    if len(ent) == 0:
                        entity.remove(ent)
                # entity.remove('')
                # entity = list(set(entity))
                tree_dic[file][mid]['entity'] = entity
    #------------------------------------------------
    pheme_files_senti = os.listdir(pheme_senti_path)
    for i in range(len(pheme_files_senti)):
        file = pheme_files_senti[i].split('.')[0]
        # file_df_senti = pd.read_csv(pheme_senti_path+file+'.csv')
        # for j in range(len(file_df_senti['mid'])):
        #     mid = file_df_senti['mid'][j]
        #     tree_dic[file][mid]['senti'] = senti_trans(file_df_senti['sentiment'][j].strip('\t'))
        # print('file name:',file)
        with open(pheme_senti_path+file+'.csv','r',encoding='utf-8')as f:
            reader = csv.reader(f)
            for line in reader:
                mid = int(line[1].strip('\t'))
                tree_dic[file][mid]['senti'] = float(line[5].strip('\t'))
    #--------------------------------------------------
    pheme_files_concept = os.listdir(pheme_concept_yago_path)
    for i in range(len(pheme_files_concept)):
        file = pheme_files_concept[i].split('.')[0]
        file_df_concept = pd.read_csv(pheme_concept_yago_path + file + '.csv')
        for j in range(len(file_df_concept['mid'])):
            mid = file_df_concept['mid'][j]
            # tree_dic[file][mid]['senti'] = senti_trans(file_df_senti['sentiment'][j])
            concept = [(' ').join(conc.strip(' ').strip('\'').strip('<').strip('>').split('_')) for conc in file_df_concept['concept'][j].strip('\t').strip('[').strip(']').split(',')]
            concept = [con.strip(' ') for con in concept]
            # concept.remove('')
            for con in concept:
                if len(con) == 0:
                    concept.remove(con)
            # concept = list(set(concept))
            tree_dic[file][mid]['concept'] = concept
    # max_t = 0
    # for file in os.listdir(pheme_entity_path):
    #     file = file.split('.')[0]
    #     for mid in tree_dic[file]:
    #         # print(tree_dic[file][mid]['parent'])
    #         # print(tree_dic[file][mid]['text'])
    #         # print(tree_dic[file][mid]['t'])
    #         # print(tree_dic[file][mid]['entity'])
    #         # print(tree_dic[file][mid]['senti'])
    #         # print(tree_dic[file][mid]['concept'])
    #         t = tree_dic[file][mid]['t']
    #         if t > max_t:
    #             max_t = t
    # print(max_t) #86386
    return tree_dic


#时间划分segment：N N=3 初始期、爆发期、衰弱期（自己想的，找资料支撑）
def time_equal_segment(sub_tree_dic):
    t_list = []
    for mid in sub_tree_dic:
        t = sub_tree_dic[mid]['t']
        t_list.append(t)
    max_t = max(t_list)
    sliding_T = int(max_t/3)
    T_num = 3
    # print('test t:',sliding_T)

    return sliding_T, T_num

def time_Kleinberg_segment():
    pass

def post_equal_segment():
    pass

#---------------图节点处理-------------------------------
def node2index():
    tree_dic = read_data()
    # files_name = os.listdir()
    files_name = [file.split('.')[0] for file in os.listdir(pheme_entity_path)]
    node_lst = []

    for file in files_name:
        for mid in tree_dic[file]:
            node_lst.append(str(mid))
    for file in files_name:
        for mid in tree_dic[file]:
            node_lst += tree_dic[file][mid]['entity']
            node_lst += tree_dic[file][mid]['concept']
    # print(node_lst[-2:-1])
    # print(type(node_lst[-2]))
    # print(len(node_lst[-2]))
    # print(node_lst[-3])
    # print(len(node_lst[-3]))
    node_lst = list(set(node_lst))
    with open('../data/weibo/' + 'node2idx_mid.txt', 'w', encoding='utf-8', newline='')as f:
        for i, node in enumerate(node_lst):
            string = node + '\t' + str(i) + '\n'
            f.writelines(string)
    with open('../data/weibo/'+'mid2text.txt','w',encoding='utf-8',newline='')as f:
        for file in files_name:
            for mid in tree_dic[file]:
                string = str(mid) +'\t' + tree_dic[file][mid]['text'] + '\n'
                f.writelines(string)

def load_node2index():
    with open('../data/weibo/' + 'node2idx_mid.txt', 'r',encoding='utf-8')as f:
        node2idx = f.readlines()
    node2idx_dict = {}
    for line in node2idx:
        # print(line)
        # print(len(line.strip('\n').split('\t')))

        node = line.strip('\n').split('\t')[0]
        idx = line.strip('\n').split('\t')[1]
        node2idx_dict[node] = idx
    return node2idx_dict

def process_node2index(node2idx, sub_tree_dic):
    node_lst = []
    text_node_lst = []
    for mid in sub_tree_dic:
        # text = sub_tree_dic[mid]['text']
        mid_new = str(mid)
        node_lst.append(node2idx[mid_new])
        text_node_lst.append(node2idx[mid_new])
        for entity in sub_tree_dic[mid]['entity']:
            node_lst.append(node2idx[entity])
        for concept in sub_tree_dic[mid]['concept']:
            node_lst.append(node2idx[concept])
    node_lst = list(set(node_lst))
    return node_lst, text_node_lst


def load_mid2label():
    label_path = '../data/weibo/weibo_id_label.txt'
    mid_label = list(open(label_path, "r",encoding='utf-8'))
    mid2label_dict = {}
    for i, m_l in enumerate(mid_label):
        # print(i)
        # print(m_l)
        # print(len(m_l.strip('\n').split('\t')))
        # print(m_l.strip('\n').split('\t'))
        mid = m_l.strip('\n').split('\t')[0]
        label = m_l.strip('\n').split('\t')[1]
        if label == 'True':
            label = 0
        elif label == 'False':
            label = 1
        mid2label_dict[mid] = int(label)
    return mid2label_dict

def load_mid2label_re():
    label_path = '../data/weibo/weibo_id_label_bigcn.txt'
    mid_label = list(open(label_path, "r",encoding='utf-8'))
    mid2label_dict = {}
    for i, m_l in enumerate(mid_label):
        # print(i)
        # print(m_l)
        # print(len(m_l.strip('\n').split('\t')))
        # print(m_l.strip('\n').split('\t'))
        mid = m_l.strip('\n').split(' ')[0]
        label = m_l.strip('\n').split(' ')[1]
        if label == '1':
            label = 0
        elif label == '0':
            label = 1
        mid2label_dict[mid] = int(label)
    return mid2label_dict

def jieba_segment_zh(stopwords, text):
    tweet_fenci = []
    # stop = set(stopwords.words('english'))

    # print(stopwords)
    # for tw in tweet:
    text = string_re_zh(text)
    sen_seg = jieba.cut(text)
    fliter_tw = [w for w in sen_seg if w not in stopwords]
    # tweet_fenci.append(fliter_tw)
    return fliter_tw

def string_re_zh(string):
    r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'  # 用户也可以在此进行自定义过滤字符
    r2 = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
    r3 = "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+"
    r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"

    # for i in range(len(tweet)):
    #     sentence = tweet[i]
    cleanr = re.compile('<.*?>')
    string = re.sub(cleanr, ' ', string)
    string = re.sub(r4, '', string)
    # tweet[i] = sentence
    return string

#--------------pmi------------------------------------
def get_window(content_lst,window_size):
    word_window_freq = defaultdict(int)  # w(i)  单词在窗口单位内出现的次数
    word_pair_count = defaultdict(int)  # w(i, j)
    windows_len = 0
    windows = list()
    length = len(content_lst)

    if length <= window_size:
        windows.append(content_lst)
    else:
        for j in range(length - window_size + 1):
            window = content_lst[j: j + window_size]
            windows.append(list(set(window)))

    for window in windows:
        for word in window:
            word_window_freq[word] += 1

        for word_pair in itertools.combinations(window, 2):
            word_pair_count[word_pair] += 1

    windows_len += len(windows)
    return word_window_freq, word_pair_count, windows_len


def cal_pmi(W_ij, W, word_freq_1, word_freq_2):
    p_i = word_freq_1 / W
    p_j = word_freq_2 / W
    p_i_j = W_ij / W
    pmi = log(p_i_j / (p_i * p_j))

    return pmi


def count_pmi(windows_len, word_pair_count, word_window_freq, threshold):
    word_pmi_lst = list()
    for word_pair, W_i_j in tqdm(word_pair_count.items(), desc="Calculate pmi between words"):
        word_freq_1 = word_window_freq[word_pair[0]]
        word_freq_2 = word_window_freq[word_pair[1]]

        pmi = cal_pmi(W_i_j, windows_len, word_freq_1, word_freq_2)
        if pmi <= threshold:
            continue
        word_pmi_lst.append([word_pair[0], word_pair[1], pmi])
    return word_pmi_lst


def get_pmi_edge(content_lst,node_idx, sub_tree_dic, window_size=20, threshold=0.):
    word_window_freq, word_pair_count, windows_len = get_window(content_lst,window_size)

    pmi_edge_lst = count_pmi(windows_len, word_pair_count, word_window_freq, threshold)
    # print("Total number of edges between word:", len(pmi_edge_lst))
    # pmi_time = time() - pmi_start
    return pmi_edge_lst

#-------------tf-idf----------------------------------
def get_tfidf_edge(entity, entity_list, sub_tree_dic):
    tf = get_tf_score(entity,entity_list)
    idf = get_idf_score(entity, sub_tree_dic)
    return tf*idf

def get_tf_score(entity, entity_list):
    tf = entity_list.count(entity)/len(entity_list)
    return tf

def get_idf_score(entity, sub_tree_dic):
    count = 0
    for mid in sub_tree_dic:
        if entity in sub_tree_dic[mid]['entity']:
            count += 1
    idf = log(len(sub_tree_dic)/(count + 1))
    return idf


#--------------建立动态图-------------------------------
def build_temporal_propagation_graph(text_node_idx,node2idx,sub_tree_dic):
    length = len(text_node_idx)
    sliding_T, T_num = time_equal_segment(sub_tree_dic)
    temporal_matrix = np.zeros((T_num,length,length),dtype=np.float)
    for mid in sub_tree_dic:
        # for i in range(T_num):
        # print('test mid:',mid)
        # print('test t:',sub_tree_dic[mid]['t'])
        # print('test sliding t:',sliding_T)
        if sub_tree_dic[mid]['t'] < sliding_T:
            idx = node2idx[str(mid)]
            new_idx = text_node_idx.index(idx)
            # parent_idx_test = node2idx[str(sub_tree_dic[mid]['parent'])]
            # print('parent idx test:',parent_idx_test)
            if str(sub_tree_dic[mid]['parent']) in node2idx:
                parent_idx = node2idx[str(sub_tree_dic[mid]['parent'])]
                new_parent_idx = text_node_idx.index(parent_idx)
                # print('solo senti 0 :',new_idx,new_parent_idx,sub_tree_dic[mid]['senti'])
                assert new_idx != new_parent_idx
                temporal_matrix[0][new_idx][new_parent_idx] = float(1.0)
                temporal_matrix[0][new_parent_idx][new_idx] = float(1.0)
                temporal_matrix[1][new_idx][new_parent_idx] = float(1.0)
                temporal_matrix[1][new_parent_idx][new_idx] = float(1.0)
                temporal_matrix[2][new_idx][new_parent_idx] = float(1.0)
                temporal_matrix[2][new_parent_idx][new_idx] = float(1.0)
            # else:
            #     print('sliding 0 sub_tree_dic[mid][parent] not in node2idx',sub_tree_dic[mid]['parent'])
        elif sub_tree_dic[mid]['t'] < 2*sliding_T and sub_tree_dic[mid]['t'] >= sliding_T:
            idx = node2idx[str(mid)]
            new_idx = text_node_idx.index(idx)
            if str(sub_tree_dic[mid]['parent']) in node2idx:
                parent_idx = node2idx[str(sub_tree_dic[mid]['parent'])]
                new_parent_idx = text_node_idx.index(parent_idx)
                assert new_idx != new_parent_idx
                # print('solo senti 1:', new_idx,new_parent_idx,sub_tree_dic[mid]['senti'])
                temporal_matrix[1][new_idx][new_parent_idx] = float(1.0)
                temporal_matrix[1][new_parent_idx][new_idx] = float(1.0)
                temporal_matrix[2][new_idx][new_parent_idx] = float(1.0)
                temporal_matrix[2][new_parent_idx][new_idx] = float(1.0)
            # else:
            #     print('sliding 1 sub_tree_dic[mid][parent] not in node2idx',sub_tree_dic[mid]['parent'])
        else:
            idx = node2idx[str(mid)]
            new_idx = text_node_idx.index(idx)
            if str(sub_tree_dic[mid]['parent']) in node2idx:
                parent_idx = node2idx[str(sub_tree_dic[mid]['parent'])]
                new_parent_idx = text_node_idx.index(parent_idx)
                # print('solo senti 2 :', new_idx,new_parent_idx,sub_tree_dic[mid]['senti'])
                assert new_idx != new_parent_idx
                temporal_matrix[2][new_idx][new_parent_idx] = float(1.0)
                temporal_matrix[2][new_parent_idx][new_idx] = float(1.0)
            # else:
            #     print('sliding 3 sub_tree_dic[mid][parent] not in node2idx',sub_tree_dic[mid]['parent'])
            #全都落在这里了
    return temporal_matrix


def build_temporal_knowledge_graph(node_idx,node2idx,sub_tree_dic):
    length = len(node_idx)
    sliding_T, T_num = time_equal_segment(sub_tree_dic)
    temporal_matrix = np.zeros((T_num, length, length),dtype=np.float32)
    content_lst_T = []
    content_lst_2T = []
    content_lst_3T = []
    #--------tfidf-----------------
    for mid in sub_tree_dic:
        # for i in range(T_num):
        if sub_tree_dic[mid]['t'] < sliding_T:
            idx = node2idx[str(mid)]
            new_idx = node_idx.index(idx)
            entity_lst = sub_tree_dic[mid]['entity']
            concept_lst = sub_tree_dic[mid]['concept']
            for entity in entity_lst:
                entity_idx = node2idx[entity]
                new_entity_idx = node_idx.index(entity_idx)
                assert new_idx != new_entity_idx
                tfidf = get_tfidf_edge(entity, entity_lst, sub_tree_dic)
                # print('test tfidf:',tfidf)
                temporal_matrix[0][new_idx][new_entity_idx] = tfidf
                # print('test tfidf____:', new_idx,new_entity_idx,temporal_matrix[0][new_idx][new_entity_idx])
                temporal_matrix[0][new_entity_idx][new_idx] = tfidf
                temporal_matrix[1][new_idx][new_entity_idx] = tfidf
                # print('test tfidf____:', new_idx,new_entity_idx,temporal_matrix[0][new_idx][new_entity_idx])
                temporal_matrix[1][new_entity_idx][new_idx] = tfidf
                temporal_matrix[2][new_idx][new_entity_idx] = tfidf
                # print('test tfidf____:', new_idx,new_entity_idx,temporal_matrix[0][new_idx][new_entity_idx])
                temporal_matrix[2][new_entity_idx][new_idx] = tfidf
                content_lst_T.append(entity)
            for concept in concept_lst:
                content_lst_T.append(concept)
            # content_lst = entity_lst+concept_lst
        elif sub_tree_dic[mid]['t'] < 2*sliding_T and sub_tree_dic[mid]['t'] >= sliding_T:
            idx = node2idx[str(mid)]
            new_idx = node_idx.index(idx)
            entity_lst = sub_tree_dic[mid]['entity']
            concept_lst = sub_tree_dic[mid]['concept']
            for entity in entity_lst:
                entity_idx = node2idx[entity]
                new_entity_idx = node_idx.index(entity_idx)
                assert new_idx != new_entity_idx
                tfidf = get_tfidf_edge(entity, entity_lst, sub_tree_dic)
                temporal_matrix[1][new_idx][new_entity_idx] = tfidf
                temporal_matrix[1][new_entity_idx][new_idx] = tfidf

                temporal_matrix[2][new_idx][new_entity_idx] = tfidf
                temporal_matrix[2][new_entity_idx][new_idx] = tfidf
                content_lst_2T.append(entity)
            for concept in concept_lst:
                content_lst_2T.append(concept)

        else:
            idx = node2idx[str(mid)]
            new_idx = node_idx.index(idx)
            entity_lst = sub_tree_dic[mid]['entity']
            concept_lst = sub_tree_dic[mid]['concept']
            for entity in entity_lst:
                entity_idx = node2idx[entity]
                new_entity_idx = node_idx.index(entity_idx)
                assert new_idx != new_entity_idx
                tfidf = get_tfidf_edge(entity, entity_lst, sub_tree_dic)
                temporal_matrix[2][new_idx][new_entity_idx] = tfidf
                temporal_matrix[2][new_entity_idx][new_idx] = tfidf
                content_lst_3T.append(entity)
            for concept in concept_lst:
                content_lst_3T.append(concept)

    pmi_edge_list_T = get_pmi_edge(content_lst_T, node_idx, sub_tree_dic, window_size=20, threshold=0.)
    pmi_edge_list_2T = get_pmi_edge(content_lst_2T, node_idx, sub_tree_dic, window_size=20, threshold=0.)
    pmi_edge_list_3T = get_pmi_edge(content_lst_3T, node_idx, sub_tree_dic, window_size=20, threshold=0.)
    for word_pair_0, word_pair_1, pmi in pmi_edge_list_T:
        idx_0 = node2idx[word_pair_0]
        new_idx_0 = node_idx.index(idx_0)
        idx_1 = node2idx[word_pair_1]
        new_idx_1 = node_idx.index(idx_1)
        # print('test pmi:', pmi)
        assert new_idx_0 != new_idx_1
        temporal_matrix[0][new_idx_0][new_idx_1] = pmi
        temporal_matrix[0][new_idx_1][new_idx_0] = pmi
        temporal_matrix[1][new_idx_0][new_idx_1] = pmi
        temporal_matrix[1][new_idx_1][new_idx_0] = pmi
        temporal_matrix[2][new_idx_0][new_idx_1] = pmi
        temporal_matrix[2][new_idx_1][new_idx_0] = pmi

    for word_pair_0, word_pair_1, pmi in pmi_edge_list_2T:
        idx_0 = node2idx[word_pair_0]
        new_idx_0 = node_idx.index(idx_0)
        idx_1 = node2idx[word_pair_1]
        new_idx_1 = node_idx.index(idx_1)
        assert new_idx_0 != new_idx_1
        temporal_matrix[1][new_idx_0][new_idx_1] = pmi
        temporal_matrix[1][new_idx_1][new_idx_0] = pmi
        temporal_matrix[2][new_idx_0][new_idx_1] = pmi
        temporal_matrix[2][new_idx_1][new_idx_0] = pmi
    for word_pair_0, word_pair_1, pmi in pmi_edge_list_3T:
        idx_0 = node2idx[word_pair_0]
        new_idx_0 = node_idx.index(idx_0)
        idx_1 = node2idx[word_pair_1]
        new_idx_1 = node_idx.index(idx_1)
        assert new_idx_0 != new_idx_1
        temporal_matrix[2][new_idx_0][new_idx_1] = pmi
        temporal_matrix[2][new_idx_1][new_idx_0] = pmi


    return temporal_matrix




def main():
    # node2index()
    node2idx_dict = load_node2index()
    # node2idx_dict = re_process_node2index()
    mid2label_dict = load_mid2label()
    tree_dic = read_data()
    files_name = [file.split('.')[0] for file in os.listdir(pheme_entity_path)]
    node_idx_final = []
    text_node_idx_final = []
    temp_propagation_graph_final = []
    temp_knowledge_graph_final = []
    label_final = []
    root_index_final = []
    for file in files_name:
        if file == '3495745049431351':
            continue
        # print_count = 0
        # for i,mid in enumerate(tree_dic[file]):
        #     if i == 0:
        #         print('all about file:',tree_dic[file][mid]['parent'])
        #         print('all about file:', tree_dic[file][mid]['text'])
        #         print('all about file:', tree_dic[file][mid]['t'])
        #         print('all about file:', tree_dic[file][mid]['entity'])
        #         print('all about file:', type(tree_dic[file][mid]['senti']))
        #         print('all about file:', tree_dic[file][mid]['concept'])
        else:
            for mid in tree_dic[file]:
                if tree_dic[file][mid]['parent'] == 'None':
                    # mid_root.append(str(mid))
                    root_idx = node2idx_dict[str(mid)]
                    root_index_final.append(root_idx)
            label_final.append(mid2label_dict[file])
            node_idx, text_node_idx = process_node2index(node2idx_dict,tree_dic[file])
            # print(node_inx)
            # print('test file:',file)
            # -------------------传播图------------------------
            # print('test text_node_idx:',text_node_idx)
            try:
                assert root_idx in text_node_idx
            except:
                with open('error_file_rootindex.txt','w',encoding='utf-8',newline='')as f:
                    string = file+'\t'+root_idx+'\n'
                    f.writelines(string)
                # print('test file:',file)
                # print(root_idx)

            try:
                temp_propagation_graph = build_temporal_propagation_graph(text_node_idx,node2idx_dict,tree_dic[file])
            except ValueError:
                with open('error_file_propagation.txt','w',encoding='utf-8',newline='')as f:
                    string = file+'\t'+text_node_idx+'\n'
                    f.writelines(string)
                # print('test file:', file)
                # print('test text_node_idx:', text_node_idx)
                # break
            # print('temp propagation graph 0',temp_propagation_graph[0].sum())
            # print('temp propagation graph 1', temp_propagation_graph[1])
            # print('temp propagation graph 2', temp_propagation_graph[2])

            temp_knowledge_graph = build_temporal_knowledge_graph(node_idx,node2idx_dict,tree_dic[file])
            # print('temp knowledge graph 0',temp_knowledge_graph[0].sum())
            print('propagation graph',temp_propagation_graph.shape)
            print('knowledge graph', temp_knowledge_graph.shape)

            text_node_idx = np.array(text_node_idx)
            node_idx = np.array(node_idx)
            node_idx_final.append(node_idx)
            text_node_idx_final.append(text_node_idx)
            propagation_s0 = sparse.csr_matrix(temp_propagation_graph[0])
            propagation_s1 = sparse.csr_matrix(temp_propagation_graph[1])
            propagation_s2 = sparse.csr_matrix(temp_propagation_graph[2])
            temp_propagation_graph_final.append([propagation_s0,propagation_s1,propagation_s2])

            knowledge_s0 = sparse.csr_matrix(temp_knowledge_graph[0])
            knowledge_s1 = sparse.csr_matrix(temp_knowledge_graph[1])
            knowledge_s2 = sparse.csr_matrix(temp_knowledge_graph[2])
            temp_knowledge_graph_final.append([knowledge_s0,knowledge_s1,knowledge_s2])

    with open(pheme_temporal_path+'propagation_node_idx.npy','wb')as f:
        text_node_idx_final = np.array(text_node_idx_final)
        np.save(f,text_node_idx_final)
    with open(pheme_temporal_path+'knowledge_node_idx.npy','wb')as f:
        node_idx_final = np.array(node_idx_final)
        np.save(f,node_idx_final)
    with open(pheme_temporal_path+'propagation_node.npy','wb')as f:
        temp_propagation_graph_final = np.array(temp_propagation_graph_final)
        np.save(f, temp_propagation_graph_final)
    with open(pheme_temporal_path+'knowledge_node.npy','wb')as f:
        temp_knowledge_graph_final = np.array(temp_knowledge_graph_final)
        np.save(f, temp_knowledge_graph_final)
    with open(pheme_temporal_path+'label.npy','wb')as f:
        label_final = np.array(label_final)
        np.save(f, label_final)
    with open(pheme_temporal_path+'propagation_root_index.npy','wb')as f:
        root_index_final = np.array(root_index_final)
        np.save(f, root_index_final)

def main_re():
    # node2index()
    node2idx_dict = load_node2index()
    # node2idx_dict = re_process_node2index()
    mid2label_dict = load_mid2label_re()
    tree_dic = read_data()
    files_name = [file.split('.')[0] for file in os.listdir(pheme_entity_path)]
    node_idx_final = []
    text_node_idx_final = []
    temp_propagation_graph_final = []
    temp_knowledge_graph_final = []
    label_final = []
    root_index_final = []
    for file in files_name:
        if file in ['3495745049431351','3843025795130981','3926120686649225','3925816377308372',\
                    '3853172374694439','3852804358130490','3852846782594165']:
            continue
        # print_count = 0
        # for i,mid in enumerate(tree_dic[file]):
        #     if i == 0:
        #         print('all about file:',tree_dic[file][mid]['parent'])
        #         print('all about file:', tree_dic[file][mid]['text'])
        #         print('all about file:', tree_dic[file][mid]['t'])
        #         print('all about file:', tree_dic[file][mid]['entity'])
        #         print('all about file:', type(tree_dic[file][mid]['senti']))
        #         print('all about file:', tree_dic[file][mid]['concept'])
        else:
            for mid in tree_dic[file]:
                if tree_dic[file][mid]['parent'] == 'None':
                    # mid_root.append(str(mid))
                    root_idx = node2idx_dict[str(mid)]
                    root_index_final.append(root_idx)
            label_final.append(mid2label_dict[file])
            node_idx, text_node_idx = process_node2index(node2idx_dict,tree_dic[file])
            # print(node_inx)
            # print('test file:',file)
            # -------------------传播图------------------------
            # print('test text_node_idx:',text_node_idx)
            try:
                assert root_idx in text_node_idx
            except:
                with open('error_file_rootindex.txt','w',encoding='utf-8',newline='')as f:
                    string = file+'\t'+root_idx+'\n'
                    f.writelines(string)
                # print('test file:',file)
                # print(root_idx)

            try:
                temp_propagation_graph = build_temporal_propagation_graph(text_node_idx,node2idx_dict,tree_dic[file])
            except ValueError:
                with open('error_file_propagation.txt','w',encoding='utf-8',newline='')as f:
                    string = file+'\t'+text_node_idx+'\n'
                    f.writelines(string)
                # print('test file:', file)
                # print('test text_node_idx:', text_node_idx)
                # break
            # print('temp propagation graph 0',temp_propagation_graph[0].sum())
            # print('temp propagation graph 1', temp_propagation_graph[1])
            # print('temp propagation graph 2', temp_propagation_graph[2])

            temp_knowledge_graph = build_temporal_knowledge_graph(node_idx,node2idx_dict,tree_dic[file])
            # print('temp knowledge graph 0',temp_knowledge_graph[0].sum())
            print('propagation graph',temp_propagation_graph.shape)
            print('knowledge graph', temp_knowledge_graph.shape)

            text_node_idx = np.array(text_node_idx)
            node_idx = np.array(node_idx)
            node_idx_final.append(node_idx)
            text_node_idx_final.append(text_node_idx)
            propagation_s0 = sparse.csr_matrix(temp_propagation_graph[0])
            propagation_s1 = sparse.csr_matrix(temp_propagation_graph[1])
            propagation_s2 = sparse.csr_matrix(temp_propagation_graph[2])
            temp_propagation_graph_final.append([propagation_s0,propagation_s1,propagation_s2])

            knowledge_s0 = sparse.csr_matrix(temp_knowledge_graph[0])
            knowledge_s1 = sparse.csr_matrix(temp_knowledge_graph[1])
            knowledge_s2 = sparse.csr_matrix(temp_knowledge_graph[2])
            temp_knowledge_graph_final.append([knowledge_s0,knowledge_s1,knowledge_s2])

    with open(pheme_temporal_path+'propagation_node_idx.npy','wb')as f:
        text_node_idx_final = np.array(text_node_idx_final)
        np.save(f,text_node_idx_final)
    with open(pheme_temporal_path+'knowledge_node_idx.npy','wb')as f:
        node_idx_final = np.array(node_idx_final)
        np.save(f,node_idx_final)
    with open(pheme_temporal_path+'propagation_node.npy','wb')as f:
        temp_propagation_graph_final = np.array(temp_propagation_graph_final)
        np.save(f, temp_propagation_graph_final)
    with open(pheme_temporal_path+'knowledge_node.npy','wb')as f:
        temp_knowledge_graph_final = np.array(temp_knowledge_graph_final)
        np.save(f, temp_knowledge_graph_final)
    with open(pheme_temporal_path+'label.npy','wb')as f:
        label_final = np.array(label_final)
        np.save(f, label_final)
    with open(pheme_temporal_path+'propagation_root_index.npy','wb')as f:
        root_index_final = np.array(root_index_final)
        np.save(f, root_index_final)


def main_tri():
    node2idx_dict = load_node2index()
    # node2idx_dict = re_process_node2index()
    # mid2label_dict = load_mid2label()
    # tree_dic = read_data()
    files_name = [file.split('.')[0] for file in os.listdir(pheme_entity_path)]
    tree_dic = read_data()
    mid_root = []
    for file in files_name:
        for mid in tree_dic[file]:
            if tree_dic[file][mid]['parent'] == 'None':
                mid_root.append(str(mid))


    root_index_final = []
    for mid in mid_root:
        root_index_final.append(node2idx_dict[mid])
    print('test root_index_final:',root_index_final)
    with open(pheme_temporal_path+'propagation_root_index.npy','wb')as f:
        root_index_final = np.array(root_index_final)
        np.save(f, root_index_final)



if __name__ == '__main__':
    node2index()
    main_re()
    # main_tri()
    # tree_dic = read_data()
    # file_len = len(tree_dic)

# read_data()

# t = 'Sat Aug 09 22:33:15 +0000 2014'
# t_init = 'Sat Aug 09 22:33:06 +0000 2014'
# print(trans_time(t, t_init))
