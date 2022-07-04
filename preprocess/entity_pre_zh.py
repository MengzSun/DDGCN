#########################################################################################
#中文微博原始数据集处理
#########################################################################################
from urllib.request import quote
import requests
import json
import csv
import os
import pandas as pd
from tqdm import tqdm

#原始数据集地址
weibo_path = '../data/weibo/Weibo_raw/'
#处理后的clean数据输出地址
weibo_output_path = '../data/weibo/weibo_clean/'
#获取帖子和评论文本的知识实体后数据输出地址
weibo_entity_path = '../data/weibo/weibo_entity/'

#读取原始微博数据json文件，提取帖子/评论mid，传播父节点，帖子/评论文本及发布时间信息，输出到weibo_output_path路径中，
#每个帖子输出一个csv文件
def read_json():
    weibo_files = os.listdir(weibo_path)
    weibo_files_num = len(weibo_files)

    for i in range(weibo_files_num):
        file_name = weibo_files[i].split('.')[0]
        file_df = pd.DataFrame(columns=['mid','parent','text','t'])
        with open(weibo_path+weibo_files[i],'r',errors="ignore", encoding="utf-8") as load_news:
            file = json.load(load_news)
            # file_df.loc[num]
            print(type(file))#list
            file_len = len(file)
        for j in range(file_len):
            mid = file[j]['mid']+'\t'
            parent = str(file[j]['parent'])+'\t'
            text = file[j]['original_text']+'\t'
            t = str(file[j]['t'])+'\t'
            file_df.loc[j] = [mid,parent,text,t]
        weibo_output = weibo_output_path+file_name+'.csv'
        file_df.to_csv(weibo_output)




    # print(tweetId)
    # print(imageId)
    # print(tweetText)
    # return tweetId, imageId, tweetText,label

#读取clean数据集中csv文件，对文本信息做entity-linking,得到知识实体list，和clean数据集中信息一起输出到weibo_entity_path路径中，
#每个帖子输出一个csv文件
def read_text():
    weibo_files = os.listdir(weibo_output_path)
    weibo_files_num = len(weibo_files)
    for i in tqdm(range(weibo_files_num)[0:1]):
        file_name = weibo_files[i].split('.')[0]
        file_df = pd.read_csv(weibo_output_path+weibo_files[i])
        file_df_new = pd.DataFrame(columns=['mid','parent','text','t','entity'])
        # print(file_df['text'][0])
        entity_all = []
        for j in range(len(file_df['text'])):
            mid = str(file_df['mid'][j]).strip('\t')+'\t'
            parent = file_df['parent'][j].strip('\t')+'\t'
            text = file_df['text'][j].strip('\t')+'\t'
            t = str(file_df['t'][j]).strip('\t')+'\t'
            entity_list =  str(entity_get(text.strip('\t')))+'\t'
            file_df_new.loc[j] = [mid, parent, text, t,entity_list]
        #     entity_all.append(entity_list)
        # file_df_new['entity'] = entity_all
        # file_df.drop('Unnamed: 0',axis=1)
        weibo_output = weibo_entity_path+file_name+'.csv'
        # weibo_output = '../data/weibo/'+file_name+'.csv'
        file_df_new.to_csv(weibo_output)

#若read_text函数中报错，排除错误后用read_unfinished_text完成其余文件的处理
def read_unfinished_text():
    weibo_files = os.listdir(weibo_output_path)
    weibo_files_num = len(weibo_files)
    weibo_entity_files = os.listdir(weibo_entity_path)
    weibo_entity_files_num = len(weibo_entity_files)
    count = 0
    for file in weibo_files:
        if file not in weibo_entity_files:
            print(file)
            count += 1
            file_name = file.split('.')[0]
            file_df = pd.read_csv(weibo_output_path + file)
            file_df_new = pd.DataFrame(columns=['mid', 'parent', 'text', 't', 'entity'])
            # print(file_df['text'][0])
            entity_all = []
            for j in range(len(file_df['text'])):
                mid = str(file_df['mid'][j]).strip('\t') + '\t'
                parent = file_df['parent'][j].strip('\t') + '\t'
                text = file_df['text'][j].strip('\t') + '\t'
                t = str(file_df['t'][j]).strip('\t') + '\t'
                entity_list = str(entity_get(text.strip('\t'))) + '\t'
                file_df_new.loc[j] = [mid, parent, text, t, entity_list]
                # entity_all.append(entity_list)
            # file_df_new['entity'] = entity_all
            # file_df.drop('Unnamed: 0',axis=1)
            weibo_output = weibo_entity_path + file_name + '.csv'
            file_df_new.to_csv(weibo_output)
    print(count)

#entity-linking函数，中文调用shuyan-tech的api
def entity_get(text):
    urlEncode = quote(text)
    url = 'http://shuyantech.com/entitylinkingapi?q='+urlEncode+'&apikey=2aeae65c1378851a7a290703f4285612'
    # print(url)
    headers = {
        'Accept' : 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            # 'application/json, text/javascript, */*; q=0.01',
        'Accepting-Encoding':'gzip, deflate, sdch',
        'Content-Type':'text/html; charset=utf-8',
        'Accept - Language': 'zh-CN,zh;q=0.8',
        'Cache-Control':'max-age=0',
        'Proxy-Connection': 'keep-alive',
        'Host':'shuyantech.com',
        # 'Referer': 'http://shuyantech.com/entitylinking',
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 SE 2.X MetaSr 1.0'
        #apikey:2aeae65c1378851a7a290703f4285612
        # 'X-Requested-With':'XMLHttpRequest'
        }
    params = {'q':text,'apikey':'2aeae65c1378851a7a290703f4285612'}
    res = requests.get(url,headers = headers,data = params)
    # print(res.status_code)
    # print(res)
    # print(type(res.text))
    # print(res.text)
    # print(type(res.json()))
    # print(res.json()['entities'])
    #------------------------------------
    entity_list = []
    for en in res.json()['entities']:
        entity_list.append(en[1])
    print(entity_list)

    return entity_list

#输出帖子对应label的txt文件
def read_file_label():
    weibo_files = os.listdir(weibo_path)
    weibo_files_num = len(weibo_files)
    with open('../data/weibo/weibo_id_label.txt', 'w', encoding='utf-8', newline='')as f:
        for i in range(weibo_files_num):
            file_name = weibo_files[i].split('.')[0]
            # file_df = pd.DataFrame(columns=['mid', 'parent', 'text', 't'])
            with open(weibo_path + weibo_files[i], 'r', errors="ignore", encoding="utf-8") as load_news:
                file = json.load(load_news)
                # file_df.loc[num]
                print(type(file))  # list
                file_len = len(file)
            label = str(file[0]['verified'])

            string = str(file_name)+'\t'+label+'\n'
            f.writelines(string)
        # for j in range(file_len):
        #     mid = file[j]['mid'] + '\t'
        #     parent = str(file[j]['parent']) + '\t'
        #     text = file[j]['original_text'] + '\t'
        #     t = str(file[j]['t']) + '\t'
        #     file_df.loc[j] = [mid, parent, text, t]
        # weibo_output = weibo_output_path + file_name + '.csv'
        # file_df.to_csv(weibo_output)

# entity_get('震惊，转发求证：【想都不敢想 ，在美国一桶金龙鱼食用油只要8元人民币】 一桶食用油相当于中国超市40多元(现在估计已经涨到五六十元了)的金龙鱼，在纽约沃尔玛感恩节时是1.6美元，圣诞节降至1.3美元。(折合人民币8.58元，而且油是绿色纯天然的，不是转基因的)，为什么中国一桶食用油要卖几十上百元？')
# entity_get('李白这首歌是在唱唐朝的李白么？')

# entitylist_list = []
# for tweet in tweetText:
#     en_list = entity_get(tweet)
#     entitylist_list.append(en_list)

# tweetId,imageId,tweetText,label = read_txt()
# with open('/home/sunmengzhu2019/kg_rumor/model_zh/text_all_entity_test.txt','w',encoding='utf-8',newline='')as f:
#     for i in range(len(tweetId)):
#         entity_list = entity_get(tweetText[i])
#         string = tweetId[i]+'\t'+tweetText[i]+'\t'+imageId[i]+'\t'+label[i]+'\t'+str(entity_list)+'\n'
#         f.write(string)

read_json()
read_text() #n*5
# read_unfinished_text()
read_file_label()


