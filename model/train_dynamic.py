import os
import sys
import time

import random
from tqdm import tqdm
import argparse
import pandas as pd

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from util import *
from path_zh import *
from data import *
from config import dual_dynamic_graph_Config, dual_dynamic_graph_Config_zh
from models import *
# from models import dual_dynamic_graph
from data_process import *
from torch_geometric.data import DataLoader
# from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

#===================train&test==================
def train_nn_nBatch_completed(train_dict, val_dict,test_dict, model, device,config,dataset,model_mode,args):
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('model bert:',model.bert)  #model.bert指bert部分的模型
    # print('model layer_stack_propagation:',model.layer_stack_propagation) #model.layer_stack_propagation指的是3个propagation GCN
    print('total number of parameters:', pytorch_total_trainable_params)
    # batch_size = 1
    # =========================================================================================
    train_x_propagation_node_indices, train_x_propagation_node_values, train_x_propagation_idx, \
    train_x_knowledge_node_indices, train_x_knowledge_node_values, train_x_knowledge_idx, \
    train_target, train_root_idx = train_dict['x_p_indices'], train_dict['x_p_values'], train_dict['idx_p'], \
                                   train_dict['x_k_indices'], train_dict['x_k_values'], train_dict['idx_k'], \
                                   train_dict['y'], train_dict['root_idx_p']
    # print('train train_x_propagation_node_values:',train_x_propagation_node_values)
    val_x_propagation_node_indices, val_x_propagation_node_values, val_x_propagation_idx, \
    val_x_knowledge_node_indices, val_x_knowledge_node_values, val_x_knowledge_idx, \
    val_target, val_root_idx = val_dict['x_p_indices'], val_dict['x_p_values'], val_dict['idx_p'], \
                               val_dict['x_k_indices'], val_dict['x_k_values'], val_dict['idx_k'], \
                               val_dict['y'], val_dict['root_idx_p']
    # t_batch = len(train_x_propagation_node)
    # batch_size
    # print('node size',train_x_propagation_node[0].size())
    # print('node type',train_x_propagation_node[0][0][0])
    traindata_list = loadData_bert(train_x_propagation_idx, train_x_propagation_node_indices,
                              train_x_propagation_node_values, \
                              train_x_knowledge_idx, train_x_knowledge_node_indices, train_x_knowledge_node_values, \
                              train_target, train_root_idx)

    valdata_list = loadData_bert(val_x_propagation_idx, val_x_propagation_node_indices, val_x_propagation_node_values, \
                            val_x_knowledge_idx, val_x_knowledge_node_indices, val_x_knowledge_node_values, \
                            val_target, val_root_idx)
    train_loader = DataLoader(traindata_list, batch_size=args.batch, shuffle=True, num_workers=5)
    val_loader = DataLoader(valdata_list, batch_size=args.batch, shuffle=True, num_workers=5)
    t_batch = len(train_loader)
    v_batch = len(val_loader)
    start_time = time.time()
    # ===============================================================================================
    model.train()
    criterion_clf = nn.BCELoss()
    #===========bert部分lr和其余部分lr相同
    # optimizer = optim.Adam(model.parameters(), lr = args.lr)#, weight_decay = config.weight_decay)
    #===========bert部分lr和其余部分lr不同
    bert_params = list(map(id,model.bert.parameters()))
    base_params = filter(lambda p:id(p) not in bert_params,model.parameters())
    # print('bert_params:',bert_params)
    # print('bert id:',id)
    optimizer = optim.Adam([
        {'params':model.bert.parameters(), 'lr':5e-5},
        {'params':base_params}
    ],lr=args.lr)#,weight_decay=config.weight_decay)
    #lr变化 scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.1,patience=5,verbose=True)
    #early stopping
    earlystopping = EarlyStopping_acc(10)
    val_loss_min = 5
    val_acc_max = 0
    #在finish predicting 前再显示一下test best result
    best_test_result_dict = {}
    best_test_result_dict['acc'] = 0
    best_test_result_dict['prec'] = 0
    best_test_result_dict['rec'] = 0
    best_test_result_dict['f1'] = 0

    for epoch in range(args.epoch):
        count_train = 0
        total_loss, total_acc = 0,0
        # count += 1
        # tqdm_train_loader = tqdm(train_loader)
        avg_acc = []
        avg_loss = []
        for Batch_data in train_loader:
            # count += 1
            Batch_data.to(device)
            optimizer.zero_grad()
            # output = model(adj_propagation,adj_knowledge,idx_propagation,idx_knowledge)
            # print('train x_propagation_node_values_1:',Batch_data.x_propagation_node_values_1)
            # print('-----------------ok5------------')
            # print('train batch:',Batch_data.batch)
            # print('train batch:',Batch_data.batch.size(),type(Batch_data.batch))
            # print('train propagation node num:',Batch_data.x_propagation_node_num)
            # print('train knowledge node num:',Batch_data.x_knowledge_node_num)
            output = model(Batch_data)
            # print('train_dynamic output:',output)
            # print('Batch data target:',Batch_data.target.size())
            # output = output.squeeze()

            loss = criterion_clf(output,Batch_data.target)
            loss.backward()
            optimizer.step()
            # correct = evaluation(output, Batch_data.target)
            # print('train max1:',torch.max(output,1)[1])
            # print('train max2:', torch.max(output, 1)[1].view(Batch_data.target.size()))
            # print('train max3:', torch.max(output, 1)[1].data)
            # print('train target:',torch.max(Batch_data.target,1)[1].data)
            correct = torch.sum(torch.eq(torch.max(output,1)[1].data, torch.max(Batch_data.target,1)[1].data)).item()
            # print('train correct:',correct)
            train_acc = correct/args.batch
            total_acc += train_acc
            # avg_acc.append(train_acc)
            total_loss += loss.item()
            # avg_loss.append(loss.item())
            #<<<<<<<<<<<<<<<<<<<<<<检查loss不变的原因<<<<<<<<<<<<<<<<<<
            # output[output >= 0.5] = 1
            # output[output < 0.5] = 0
            # if count_train == 0:
            #     output_all = output
            #     label_all = Batch_data.target
            # else:
            #     output_all = torch.cat([output_all, output], dim=0)
            #     label_all = torch.cat([label_all, Batch_data.target], dim=0)
            count_train += 1
            # if count_train%200 == 0:
            #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            print('Epoch{} | Batch{} | Train_Loss {:.4f} | Train_Accuracy {:.4f}'.format((epoch+1), \
                                        count_train, loss.item(),train_acc*100))
            # print('process step:',i,'output:',output)
            # if count%1000 == 0:
            #     print('[Epoch{}|{}]'.format(epoch+1,count),'Train | Loss:{:.5f} Acc:{:.3f}'.format(total_loss / count, total_acc / count * 100))
        print('\n [Epoch{}]'.format(epoch + 1),'Train | Loss:{:.5f} Acc:{:.3f}'.format(total_loss / t_batch,\
                                                                total_acc / t_batch * 100))
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # acc = accuracy(output_all, label_all)
        # f1, precision, recall, f1_real, precision_real, recall_real, f1_fake, precision_fake, recall_fake = \
        #     macro_f1(output_all, label_all, num_classes=2)
        # print('----------------------------------------------------')
        # print('acc:', acc, 'prec:', precision, 'rec:', recall, 'f1:', f1)
        # print('prec-fake:', precision_fake, 'rec-fake:', recall_fake, 'f1-fake:', f1_fake)
        # print('prec-real:', precision_real, 'rec-real:', recall_real, 'f1-real:', f1_real)
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        model.eval()
        with torch.no_grad():
            total_loss,total_acc = 0,0
            count_val = 0
            # tqdm_val_loader = tqdm(val_loader)
            # v_batch = len(val_x_propagation_node)
            for Batch_data in val_loader:
                Batch_data.to(device)
                output = model(Batch_data)
                # output = output.squeeze()
                loss = criterion_clf(output, Batch_data.target)
                # correct = evaluation(output, Batch_data.target)
                # correct = (torch.max(output, 1)[1].view(Batch_data.target.size()).data == Batch_data.target.data).sum()
                correct = torch.sum(torch.eq(torch.max(output,1)[1].data, torch.max(Batch_data.target,1)[1].data)).item()
                val_acc = correct / args.batch
                # total_acc += train_acc
                total_acc += val_acc
                total_loss += loss.item()
                #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                # output[output >= 0.5] = 1
                # output[output < 0.5] = 0
                # if count_val == 0:
                #     output_all = output
                #     label_all = Batch_data.target
                # else:
                #     output_all = torch.cat([output_all, output], dim=0)
                #     label_all = torch.cat([label_all, Batch_data.target], dim=0)
                count_val += 1
                #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            print("valid | Loss:{:.5f} Acc:{:.3f} \n".format(total_loss / v_batch, total_acc / v_batch * 100))
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # acc = accuracy(output_all, label_all)
            # f1, precision, recall, f1_real, precision_real, recall_real, f1_fake, precision_fake, recall_fake = \
            #     macro_f1(output_all, label_all, num_classes=2)
            # print('----------------------------------------------------')
            # print('acc:', acc, 'prec:', precision, 'rec:', recall, 'f1:', f1)
            # print('prec-fake:', precision_fake, 'rec-fake:', recall_fake, 'f1-fake:', f1_fake)
            # print('prec-real:', precision_real, 'rec-real:', recall_real, 'f1-real:', f1_real)
            #==========================================================================================
            # val loss作为标准保存模型<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # val_loss = total_loss/v_batch
            # if val_loss < val_loss_min:
            #     val_loss_min = val_loss
            #     earlystopping(total_loss / v_batch, model)
            #     test_result_dict = test_nn_nBatch_completed(test_dict,model,device,config,dataset,model_mode,args)
            #     # if test_result_dict['acc'] > best_test_result_dict['acc']:
            #     best_test_result_dict['acc'] = test_result_dict['acc']
            #     best_test_result_dict['f1'] = test_result_dict['f1']
            #     best_test_result_dict['prec'] = test_result_dict['prec']
            #     best_test_result_dict['rec'] = test_result_dict['rec']
            # val acc作为标准保存模型<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            #=========================================================================================
            val_acc = total_acc / v_batch
            if val_acc > val_acc_max:
                val_acc_max = val_acc
                earlystopping(total_acc / v_batch, model)
                test_result_dict = test_nn_nBatch_completed(test_dict, model, device, config, dataset,
                                                            model_mode,args)
                # if test_result_dict['acc'] > best_test_result_dict['acc']:
                best_test_result_dict['acc'] = test_result_dict['acc']
                best_test_result_dict['f1'] = test_result_dict['f1']
                best_test_result_dict['prec'] = test_result_dict['prec']
                best_test_result_dict['rec'] = test_result_dict['rec']


        model.train()
        scheduler.step(val_acc)
        # scheduler.step(val_acc)
    end_time = time.time()
    return best_test_result_dict, start_time, end_time


def test_nn_nBatch_completed(test_dict, model, device,config,dataset,model_mode,args):
    model.eval()
    ret_output = []
    test_result_dict = {}
    with torch.no_grad():
        count = 0
        total_acc = 0
        test_x_propagation_node_indices, test_x_propagation_node_values, test_x_propagation_idx, \
        test_x_knowledge_node_indices, test_x_knowledge_node_values, test_x_knowledge_idx, \
        test_target, test_root_idx = test_dict['x_p_indices'], test_dict['x_p_values'], test_dict['idx_p'], \
                                   test_dict['x_k_indices'], test_dict['x_k_values'], test_dict['idx_k'], \
                                   test_dict['y'], test_dict['root_idx_p']
        testdata_list = loadData_bert(test_x_propagation_idx, test_x_propagation_node_indices,
                                  test_x_propagation_node_values, \
                                  test_x_knowledge_idx, test_x_knowledge_node_indices, test_x_knowledge_node_values, \
                                  test_target, test_root_idx)
        test_loader = DataLoader(testdata_list, batch_size=args.batch, shuffle=False, num_workers=5)
        te_batch = len(test_loader)
        for Batch_data in test_loader:
            # count += 1
            Batch_data.to(device)
            output = model(Batch_data)
            # correct = (torch.max(output, 1)[1].view(Batch_data.target.size()).data == Batch_data.target.data).sum()
            # test_acc = correct/config.batch_size
            # total_acc += test_acc
            # output = output.squeeze()
            # loss = criterion_clf(output, Batch_data.target)

            # correct = (torch.max(output, 1)[1].data == torch.max(Batch_data.target, 1)[1].data).sum()
            # output[output >= 0.5] = 1
            # output[output < 0.5] = 0
            # correct = torch.sum(torch.eq(torch.max(output, 1)[1].data, torch.max(Batch_data.target, 1)[1].data)).item()
            output = torch.max(output, 1)[1].data
            target_new = torch.max(Batch_data.target, 1)[1].data
            if count == 0:
                output_all = output
                label_all = target_new
            else:
                output_all = torch.cat([output_all,output],dim=0)
                label_all = torch.cat([label_all,target_new],dim=0)
            count += 1
        # acc = accuracy(output_all,label_all)
        # f1, precision, recall, f1_real, precision_real, recall_real, f1_fake, precision_fake, recall_fake = \
        #     macro_f1(output_all, label_all, num_classes=2)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        acc = accuracy(output_all, label_all)
        f1, precision, recall, f1_real, precision_real, recall_real, f1_fake, precision_fake, recall_fake = \
            macro_f1(output_all, label_all, num_classes=2)
        print('----------------------------------------------------')
        print('acc:', acc, 'prec:', precision, 'rec:', recall, 'f1:', f1)
        print('prec-fake:', precision_fake, 'rec-fake:', recall_fake, 'f1-fake:', f1_fake)
        print('prec-real:', precision_real, 'rec-real:', recall_real, 'f1-real:', f1_real)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        with open('result_test_{}_{}.txt'.format(dataset,model_mode),'a',encoding='utf-8',newline='')as f:
            string1 = 'acc:'+str(acc)+'\t'+'f1:'+str(f1)+'\t'+'precision:'+str(precision)+'\t'+'recall:'+str(recall)+'\n'
            string2 = 'f1_real:' + str(f1_real) + '\t' + 'precision_real:' + str(
                precision_real) + '\t' + 'recall_real:' + str(recall_real)+'\n'
            string3 = 'f1_fake:' + str(f1_fake) + '\t' + 'precision_fake:' + str(
                precision_fake) + '\t' + 'recall_fake:' + str(recall_fake)+'\n\n\n'
            f.writelines(string1)
            f.writelines(string2)
            f.writelines(string3)
    test_result_dict['acc'] = acc
    test_result_dict['prec'] = precision
    test_result_dict['rec'] = recall
    test_result_dict['f1'] = f1

    # test_dict['acc'] = acc
    test_result_dict['prec_fake'] = precision_fake
    test_result_dict['rec_fake'] = recall_fake
    test_result_dict['f1_fake'] = f1_fake

    test_result_dict['prec_real'] = precision_real
    test_result_dict['rec_real'] = recall_real
    test_result_dict['f1_real'] = f1_real

    return test_result_dict
#====================main=======================
def main_nn_nBatch(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = args.dataset
    if dataset == 'pheme':
        config = dual_dynamic_graph_Config()
    elif dataset == 'weibo':
        # device = "cpu"
        # device = torch.device("cpu")
        config = dual_dynamic_graph_Config_zh()

    model_mode = args.model
    pathset = path_Set_BERT(dataset)
    # load data
    # train_dict, val_dict, test_dict = load_sparse_temporal_data(config.text_max_length, config.train, config.val, config.test,config)
    if model_mode in ['propagation','knowledge','completed_wotf','static','completed','completed_1','completed_2','bert','completed_attentionTF','completed_wobert']:
        data_process = data_process_nn_nBatch_BERT(config.text_max_length,pathset,dataset)
        # token_dict 检索entity、concept、post节点对应的nn.embedding序号，mid2regtextidx_dict检索文本对应的定长序号序列
        train_dict, val_dict, test_dict, token_dict, mid2bert_tokenizer = data_process.load_sparse_temporal_data(
            config.train, config.val, config.test)
    elif model_mode in ['completed_gru']:
        data_process = data_process_nn_nBatch_BERT_gru(config.text_max_length,pathset,dataset)
        train_dict, val_dict, test_dict, token_dict, mid2bert_tokenizer, mid2regtextidx_dict = data_process.load_sparse_temporal_data(
            config.train, config.val, config.test)
    elif model_mode in ['completed_gru_wobert','gru']:
        data_process = data_process_nn_nBatch_BERT_gru_wobert(config.text_max_length, pathset, dataset)
        train_dict, val_dict, test_dict, token_dict, mid2regtextidx_dict = data_process.load_sparse_temporal_data(
            config.train, config.val, config.test)

    # token_dict 检索entity对应的nn.embedding序号，mid2regtextidx_dict检索文本对应的定长序号序列
    # train_dict, val_dict, test_dict, token_dict, mid2regtextidx_dict = data_process.load_sparse_temporal_data(
    #     config.train, config.val, config.test)
    print("token dict len",len(token_dict))
    # print('x_p',train_dict['x_p'])
    #------------------train---------------------------
    if model_mode == 'propagation':
        model = dual_DynamicGCN_nBatch_propagation(token_dict=token_dict, mid2bert_tokenizer=mid2bert_tokenizer,
                                                 bert_path=pathset.path_bert,n_output=config.n_class, config=config, \
                                            device=device, n_hidden=3,dropout=0.2, instance_norm=False)
        # if torch.cuda.device_count() > 1:
        #     model = torch.nn.DataParallel(model)
        # #--------------------------------------------------------------------------
        # train_nn_nBatch_completed(train_dict,val_dict,test_dict,model,device,config,dataset)
        # print("finish predicting")

    elif model_mode == 'knowledge':
        model = dual_DynamicGCN_nBatch_knowledge(token_dict=token_dict, mid2bert_tokenizer=mid2bert_tokenizer,
                                                 bert_path=pathset.path_bert,n_output=config.n_class, config=config, \
                                            device=device, n_hidden=3,dropout=0.2, instance_norm=False)
        # if torch.cuda.device_count() > 1:
        #     model = torch.nn.DataParallel(model)
        # # --------------------------------------------------------------------------
        # train_nn_nBatch_completed(train_dict, val_dict, test_dict, model, device, config, dataset)
        # print("finish predicting")

    elif model_mode == 'completed_wotf':
        model = dual_DynamicGCN_nBatch_completed_wotf(token_dict=token_dict, mid2bert_tokenizer=mid2bert_tokenizer,
                                                 bert_path=pathset.path_bert,n_output=config.n_class, config=config, \
                                            device=device, n_hidden=3,dropout=0.2, instance_norm=False)
        # model = model.to(device)
        # if torch.cuda.device_count() > 1:
        #     model = torch.nn.DataParallel(model)
        # # --------------------------------------------------------------------------
        # train_nn_nBatch_completed(train_dict, val_dict, test_dict, model, device, config, dataset)
        # print("finish predicting")

    elif model_mode == 'static':
        model = dual_DynamicGCN_nBatch_completed_static(token_dict=token_dict, mid2bert_tokenizer=mid2bert_tokenizer,
                                                 bert_path=pathset.path_bert,n_output=config.n_class, config=config, \
                                            device=device, n_hidden=3,dropout=0.2, instance_norm=False)
        # model = model.to(device)
        # if torch.cuda.device_count() > 1:
        #     model = torch.nn.DataParallel(model)
        # # --------------------------------------------------------------------------
        # train_nn_nBatch_completed(train_dict, val_dict, test_dict, model, device, config, dataset)
        # print("finish predicting")

    elif model_mode == 'completed':
        model = dual_DynamicGCN_nBatch_completed(token_dict=token_dict, mid2bert_tokenizer=mid2bert_tokenizer,
                                                 bert_path=pathset.path_bert,n_output=config.n_class, config=config, \
                                            device=device, n_hidden=3,dropout=0.2, instance_norm=False)
        # model = model.to(device)
        # if torch.cuda.device_count() > 1:
        #     model = torch.nn.DataParallel(model)
        # # --------------------------------------------------------------------------
        # train_nn_nBatch_completed(train_dict, val_dict, test_dict, model, device, config, dataset)
        # print("finish predicting")
    elif model_mode == 'completed_1':
        model = dual_DynamicGCN_nBatch_completed_1(token_dict=token_dict, mid2bert_tokenizer=mid2bert_tokenizer,
                                                   bert_path=pathset.path_bert, n_output=config.n_class, config=config, \
                                                   device=device, n_hidden=3, dropout=0.2, instance_norm=False)
    elif model_mode == 'completed_2':
        model = dual_DynamicGCN_nBatch_completed_2(token_dict=token_dict, mid2bert_tokenizer=mid2bert_tokenizer,
                                                   bert_path=pathset.path_bert, n_output=config.n_class, config=config, \
                                                   device=device, n_hidden=3, dropout=0.2, instance_norm=False)

    elif model_mode == 'completed_wobert':
        model = dual_DynamicGCN_nBatch_completed_wobert(token_dict=token_dict, mid2bert_tokenizer=mid2bert_tokenizer,
                                                 bert_path=pathset.path_bert, n_output=config.n_class, config=config, \
                                                 device=device, n_hidden=3, dropout=0.2, instance_norm=False)

    elif model_mode == 'bert':
        model = dual_DynamicGCN_nBatch_BERT(token_dict=token_dict, mid2bert_tokenizer=mid2bert_tokenizer,
                                                 bert_path=pathset.path_bert,n_output=config.n_class, config=config, \
                                            device=device, n_hidden=3,dropout=0.2, instance_norm=False)
        # model = model.to(device)
        # if torch.cuda.device_count() > 1:
        #     model = torch.nn.DataParallel(model)
        # # --------------------------------------------------------------------------
        # train_nn_nBatch_completed(train_dict, val_dict, test_dict, model, device, config, dataset)
        # print("finish predicting")
    elif model_mode == 'completed_attentionTF':
        model = dual_DynamicGCN_nBatch_completed_attentionTF(token_dict=token_dict, mid2bert_tokenizer=mid2bert_tokenizer,
                                                 bert_path=pathset.path_bert,n_output=config.n_class, config=config, \
                                            device=device, n_hidden=3,dropout=0.2, instance_norm=False)
    elif model_mode == 'completed_gru_wobert':
        model = dual_DynamicGCN_nBatch_completed_gru_wobert(token_dict=token_dict, mid2regtextidx_dict=mid2regtextidx_dict,
                                                 bert_path=pathset.path_bert,n_output=config.n_class, config=config, \
                                            device=device, n_hidden=3,dropout=0.2, instance_norm=False)
    elif model_mode == 'completed_gru':
        model = dual_DynamicGCN_nBatch_completed_gru(token_dict=token_dict, mid2regtextidx_dict=mid2regtextidx_dict,\
                                                     mid2bert_tokenizer=mid2bert_tokenizer,\
                                                 bert_path=pathset.path_bert,n_output=config.n_class, config=config, \
                                            device=device, n_hidden=3,dropout=0.2, instance_norm=False)

    elif model_mode == 'gru':
        model = dual_DynamicGCN_nBatch_gru(token_dict=token_dict, mid2regtextidx_dict=mid2regtextidx_dict,
                                                 bert_path=pathset.path_bert,n_output=config.n_class, config=config, \
                                            device=device, n_hidden=3,dropout=0.2, instance_norm=False)

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    # --------------------------------------------------------------------------
    best_test_result_dict,start_time, end_time = train_nn_nBatch_completed(train_dict, val_dict, test_dict, model, device, config, dataset,model_mode,args)
    print('---------------------final best-------------------------------')
    print('parameter:','batch_size:',args.batch,'epoch_num:',args.epoch,'learning_rate:',args.lr)
    print('acc:', best_test_result_dict['acc'], 'prec:', best_test_result_dict['prec'],\
          'rec:', best_test_result_dict['rec'], 'f1:', best_test_result_dict['f1'])
    print("finish predicting")
    print(f"the running time is:,{end_time-start_time} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='dualDynamicGCN')
    parser.add_argument('--dataset',type=str,default='pheme')
    parser.add_argument('--model',type=str,default='propagation')#knowledge/completed
    parser.add_argument('--cuda',type=int,default=0)
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--epoch',type=int,default=5)
    parser.add_argument('--lr',type=float,default=5e-5)
    args = parser.parse_args()

    main_nn_nBatch(args)