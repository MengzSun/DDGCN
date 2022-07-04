#probase&YAGO
#probase直接调API
from urllib.request import quote
import requests
import json
import csv
import os
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import defaultdict
import random
from time import sleep
from threading import Thread

entity_path_zh = '../data/weibo/weibo_entity/'
concept_path_zh = '../data/weibo/weibo_concept/'
entity_path_en = '../data/pheme/pheme_entity/'
concept_path_en_yago = '../data/pheme/pheme_concept_yago/'
concept_path_zh_yago = '../data/weibo/weibo_concept_yago/'

#英文实体 probase知识图谱isA relation api调用
def en_probase_search(instance):
    url = 'https://concept.research.microsoft.com/api/Concept/ScoreByProb?instance='+instance+'&topK=10'
    # print(url)
    headers = {
        'authority':'concept.research.microsoft.com',
        'method':'GET',
        'path':'/api/Concept/ScoreByProb?instance=microsoft&topK=10',
        'scheme':'https',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        # 'application/json, text/javascript, */*; q=0.01',
        'accepting-encoding': 'gzip, deflate, br',
        # 'Content-Type': 'text/html; charset=utf-8',
        'accept - language': 'zh-CN,zh;q=0.8',
        'cache-control': 'max-age=0',
        # 'Proxy-Connection': 'keep-alive',
        # 'Host': 'shuyantech.com',
        # 'Referer': 'http://shuyantech.com/entitylinking',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36'
        # apikey:2aeae65c1378851a7a290703f4285612
        # 'X-Requested-With':'XMLHttpRequest'
        # 'cookie':"MC1=GUID=f0c0d2112b14462f85592b5ea780efa2&HASH=f0c0&LV=202009&V=4&LU=1600070436998; _cs_c=1; _mkto_trk=id:157-GQE-382&token:_mch-microsoft.com-1600157175060-17680; WRUID=2950541741441273; LPVID=YzNWE1Nzc1NTBiNjRlZjQ4; MUID=0A25163DF106601405631B77F506618A; AMCV_EA76ADE95776D2EC7F000101%40AdobeOrg=1585540135%7CMCIDTS%7C18521%7CMCMID%7C74582614025034580863195402461329685369%7CMCAAMLH-1600763166%7C11%7CMCAAMB-1600763166%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCCIDH%7C318347693%7CMCOPTOUT-1600165566s%7CNONE%7CMCAID%7CNONE%7CvVersion%7C4.4.0; mbox=session#d19358e6d19d4559890a804d63c861c2#1600159023|PC#d19358e6d19d4559890a804d63c861c2.38_0#1663403168; _cs_id=04141171-7c0c-aa64-d294-09215a7fe2c1.1600157175.1.1600158369.1600157175.1594299326.1634321175026.Lax.0; __CT_Data=gpv=5&ckp=tld&dm=microsoft.com&apv_1067_www32=5&cpv_1067_www32=5&rpv_1067_www32=4; ctm={'pgv':1294030351805486|'vst':1245646212560133|'vstr':5056047842946551|'intr':1600158370568|'v':1}; _fbp=fb.1.1621492388171.490782624; _pk_ref.1.8ebd=%5B%22%22%2C%22%22%2C1621585281%2C%22https%3A%2F%2Fwww.microsoft.com%2F%22%5D; _pk_id.1.8ebd=b8e6805073ef47ff.1621493097.3.1621585281.1621585281.; _pk_ses.1.8ebd=*; MSCC=NR; MS0=a9b66297cfa841348aef2765c2824464"
    }
    cookie = {
        'cookie': "MC1=GUID=f0c0d2112b14462f85592b5ea780efa2&HASH=f0c0&LV=202009&V=4&LU=1600070436998; _cs_c=1; _mkto_trk=id:157-GQE-382&token:_mch-microsoft.com-1600157175060-17680; WRUID=2950541741441273; LPVID=YzNWE1Nzc1NTBiNjRlZjQ4; MUID=0A25163DF106601405631B77F506618A; AMCV_EA76ADE95776D2EC7F000101%40AdobeOrg=1585540135%7CMCIDTS%7C18521%7CMCMID%7C74582614025034580863195402461329685369%7CMCAAMLH-1600763166%7C11%7CMCAAMB-1600763166%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCCIDH%7C318347693%7CMCOPTOUT-1600165566s%7CNONE%7CMCAID%7CNONE%7CvVersion%7C4.4.0; mbox=session#d19358e6d19d4559890a804d63c861c2#1600159023|PC#d19358e6d19d4559890a804d63c861c2.38_0#1663403168; _cs_id=04141171-7c0c-aa64-d294-09215a7fe2c1.1600157175.1.1600158369.1600157175.1594299326.1634321175026.Lax.0; __CT_Data=gpv=5&ckp=tld&dm=microsoft.com&apv_1067_www32=5&cpv_1067_www32=5&rpv_1067_www32=4; ctm={'pgv':1294030351805486|'vst':1245646212560133|'vstr':5056047842946551|'intr':1600158370568|'v':1}; _fbp=fb.1.1621492388171.490782624; _pk_ref.1.8ebd=%5B%22%22%2C%22%22%2C1621585281%2C%22https%3A%2F%2Fwww.microsoft.com%2F%22%5D; _pk_id.1.8ebd=b8e6805073ef47ff.1621493097.3.1621585281.1621585281.; _pk_ses.1.8ebd=*; MSCC=NR; MS0=a9b66297cfa841348aef2765c2824464"
    }
    params = {'instance': instance, 'topK': '10'}
    res = requests.get(url, headers=headers,params = params,cookies = cookie)
    print(res.status_code)
    print(res)
    print(type(res))
    # print(res.text)
    # print(type(res.json()))
    # print(res.json()['entities'])
    # ------------------------------------

# en_probase_search('microsoft')
#中文实体 probase知识图谱isA relation shuyan api调用
def cn_probase_search(instance, ip_list):
    url = 'http://shuyantech.com/api/cnprobase/concept?q='+instance+'&apikey=2aeae65c1378851a7a290703f4285612'
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        # 'application/json, text/javascript, */*; q=0.01',
        'Accepting-Encoding': 'gzip, deflate, sdch',
        'Content-Type': 'text/html; charset=utf-8',
        'Accept - Language': 'zh-CN,zh;q=0.8',
        'Cache-Control': 'max-age=0',
        'Proxy-Connection': 'keep-alive',
        'Host': 'shuyantech.com',
        # 'Referer': 'http://shuyantech.com/entitylinking',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 SE 2.X MetaSr 1.0',
        'apikey':'2aeae65c1378851a7a290703f4285612'
        # 'X-Requested-With':'XMLHttpRequest'
    }
    params = {'q': instance, 'apikey':'2aeae65c1378851a7a290703f4285612'}
    ip_new = [ip for ip in ip_list]
    for ip in ip_list:
        proxies = {
            'https':ip,
            'http':ip
        }
        try:
            res = requests.get(url, headers=headers,params = params,proxies = proxies, verify = False, timeout = 20)
            # print(res.status_code)
            entity_list = []
            for tum in res.json()['ret']:
                # entity_list.append(tum[0])
                # print(type(list(tum)))
                # print(tum)
                entity_list.append(tum[0])
            print(entity_list)
            return entity_list, ip_new
            # break
        except:
            ip_new.remove(ip)
            if len(ip_new) < 3:
                get_ip()
                ip_list += read_ip()
                ip_new += read_ip()
            continue
    # print(res)
    # print(type(res.text))
    # print(res.text)
    # print(type(res.json()['ret']))
    # print(res.json())
    # entity_list = []
    # for tum in res.json()['ret']:
    #     # entity_list.append(tum[0])
    #     # print(type(list(tum)))
    #     # print(tum)
    #     entity_list.append(tum[0])
    # print(entity_list)
    # return entity_list, ip_new

def cn_probase_search_sleep(instance):
    url = 'http://shuyantech.com/api/cnprobase/concept?q=' + instance+'&apikey=2aeae65c1378851a7a290703f4285612'
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        # 'application/json, text/javascript, */*; q=0.01',
        'Accepting-Encoding': 'gzip, deflate, sdch',
        'Content-Type': 'text/html; charset=utf-8',
        'Accept - Language': 'zh-CN,zh;q=0.8',
        'Cache-Control': 'max-age=0',
        'Proxy-Connection': 'keep-alive',
        'Host': 'shuyantech.com',
        # 'Referer': 'http://shuyantech.com/entitylinking',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 SE 2.X MetaSr 1.0'
        # apikey:2aeae65c1378851a7a290703f4285612
        # 'X-Requested-With':'XMLHttpRequest'
    }
    params = {'q': instance, 'apikey':'2aeae65c1378851a7a290703f4285612'}
    res = requests.get(url, headers=headers, data=params)

    entity_list = []
    for tum in res.json()['ret']:
        # entity_list.append(tum[0])
        # print(type(list(tum)))
        # print(tum)
        entity_list.append(tum[0])
    print(entity_list)
    return entity_list

#英文实体 yago知识图谱 isA relation获取
def en_YAGO_search():
    pheme_files = os.listdir(entity_path_en)
    pheme_files_finish = os.listdir(concept_path_en_yago)
    isA_dict = read_tsv()
    for file in pheme_files:
        if not file in pheme_files_finish:
            file_df = pd.read_csv(entity_path_en+file)
            file_df_new = pd.DataFrame(columns = ['mid','t','entity','concept'])
            for i in range(len(file_df['entity'])):
                mid = str(file_df['mid'][i]).strip('\t') + '\t'
                t = str(file_df['t'][i]).strip('\t') + '\t'
                entity_list = str(file_df['entity'][i].strip('\t')) + '\t'
                entity_isA_temp = []
                entity_lst = [ent.strip(' ').strip('\'') for ent in
                              file_df['entity'][i].strip('\t').strip('[').strip(']').split(',')]
                # print(entity_lst[0])
                # cn_probase_search(entity_lst[0])
                for entity in entity_lst:
                    ent = '_'.join(entity.split(' '))
                    ent = '<'+ent+'>'
                    if ent in isA_dict:
                        entity_isA_temp += isA_dict[ent]
                # print(entity_isA_temp)
                entity_isA_new = []
                for concept in entity_isA_temp:
                    concept_new = ' '.join(concept.strip('<').strip('>').split('_'))
                    entity_isA_new.append(concept_new)
                entity_isA_new = str(entity_isA_new)+'\t'
                file_df_new.loc[i] = [mid, t, entity_list, entity_isA_temp]
                print(entity_isA_new)

            pheme_output = concept_path_en_yago + file
            file_df_new.to_csv(pheme_output)
#中文实体 yago知识图谱isA relation获取
def cn_YAGO_search():
    weibo_files = os.listdir(entity_path_zh)
    weibo_files_finish = os.listdir(concept_path_zh_yago)
    isA_dict = read_tsv()
    for j in range(len(weibo_files)):
        file = weibo_files[j]
        if not file in weibo_files_finish:
            file_df = pd.read_csv(entity_path_zh + file)
            file_df_new = pd.DataFrame(columns=['mid', 't', 'entity', 'concept'])
            for i in range(len(file_df['entity'])):
                mid = str(file_df['mid'][i]).strip('\t') + '\t'
                t = str(file_df['t'][i]).strip('\t') + '\t'
                entity_list = str(file_df['entity'][i].strip('\t')) + '\t'
                entity_isA_temp = []
                entity_lst = [ent.strip(' ').strip('\'') for ent in
                              file_df['entity'][i].strip('\t').strip('[').strip(']').split(',')]
                # print(entity_lst[0])
                # cn_probase_search(entity_lst[0])
                for entity in entity_lst:
                    ent = '_'.join(entity.split('(')[0].strip(' ').split(' '))
                    ent = '<' + ent + '>'
                    if ent in isA_dict:
                        entity_isA_temp += isA_dict[ent]
                # print(entity_isA_temp)
                entity_isA_new = []
                for concept in entity_isA_temp:
                    concept_new = ' '.join(concept.strip('<').strip('>').split('_'))
                    entity_isA_new.append(concept_new)
                entity_isA_new = str(entity_isA_new) + '\t'
                file_df_new.loc[i] = [mid, t, entity_list, entity_isA_temp]
                print(entity_isA_new)

            weibo_output = concept_path_zh_yago + file
            file_df_new.to_csv(weibo_output)

    # weibo_files = os.listdir(entity_path_zh)

def main_isA_en():
    pass

#中文isA relation获取主函数
def main_isA_zh():
    ip_list = read_ip()
    # ip_new = []
    # for ip in ip_list:
    #     ip_new.append(ip)
    weibo_files = os.listdir(entity_path_zh)
    # def load_cn_probase(file, ip_list):
    for j in tqdm(range(len(weibo_files))):
        file = weibo_files[j]
        weibo_files_finish = os.listdir(concept_path_zh)
        if not file in weibo_files_finish:
            file_df = pd.read_csv(entity_path_zh+file)
            file_df_new = pd.DataFrame(columns=['mid','concept'])
            # print(file_df['entity'][0])
            # entity_isA_all = []
            for i in range(len(file_df['entity'])):
                mid = str(file_df['mid'][i]).strip('\t') + '\t'
                entity_isA_temp = []
                entity_lst = [ent.strip(' ').strip('\'') for ent in file_df['entity'][i].strip('\t').strip('[').strip(']').split(',')]
                # print(entity_lst[0])
                # cn_probase_search(entity_lst[0])
                for entity in entity_lst:
                    isA_concept,ip_new = cn_probase_search(entity, ip_list)
                    ip_list = ip_new
                    entity_isA_temp += isA_concept
                    r_time = random.randint(1, 5)
                    sleep(r_time)
                print('all:',entity_isA_temp)
                entity_isA = str(entity_isA_temp) + '\t'
                file_df_new.loc[i] = [mid, entity_isA]
            weibo_output = '../data/weibo/weibo_concept/' + file
            file_df_new.to_csv(weibo_output)

    # ip_list = read_ip()
    # # ip_new = []
    # # for ip in ip_list:
    # #     ip_new.append(ip)
    # weibo_files = os.listdir(entity_path_zh)
    # weibo_files_finish = os.listdir(concept_path_zh)
    # Parallel(n_jobs=30, backend='threading')(
    #     delayed(load_cn_probase)(file, ip_list) for file in tqdm(weibo_files))

#中文isA relation获取多线程函数
def cn_probase_multiprocessing(weibo_files, ip_list):
    entity2concept_dict = {}
    for j in tqdm(range(len(weibo_files))):
        file = weibo_files[j]
        weibo_files_finish = os.listdir(concept_path_zh)
        if not file in weibo_files_finish:
            file_df = pd.read_csv(entity_path_zh+file)
            file_df_new = pd.DataFrame(columns=['mid','concept'])
            # print(file_df['entity'][0])
            # entity_isA_all = []
            for i in range(len(file_df['entity'])):
                mid = str(file_df['mid'][i]).strip('\t') + '\t'
                entity_isA_temp = []
                entity_lst = [ent.strip(' ').strip('\'') for ent in file_df['entity'][i].strip('\t').strip('[').strip(']').split(',')]
                # print(entity_lst[0])
                # cn_probase_search(entity_lst[0])
                for entity in entity_lst:
                    if entity in entity2concept_dict:
                        entity_isA_temp += entity2concept_dict[entity]
                    else:
                        isA_concept,ip_new = cn_probase_search(entity, ip_list)
                        ip_list = ip_new
                        entity2concept_dict[entity] = isA_concept
                        entity_isA_temp += isA_concept
                        r_time = random.randint(1, 5)
                        sleep(r_time)
                print('all:',entity_isA_temp)
                entity_isA = str(entity_isA_temp) + '\t'
                file_df_new.loc[i] = [mid, entity_isA]
            weibo_output = '../data/weibo/weibo_concept/' + file
            file_df_new.to_csv(weibo_output)

class MyThread(Thread):
    def __init__(self,func,args):
        super(MyThread,self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.func(*self.args)

    # def get_result(self):
    #     try:
    #         return self.entity_lst,self.ip_new
    #     except Exception:
    #         return None
#中文isA relation获取多线程主函数
def main_isA_zh_multiprocessing():
    ip_list = read_ip()
    # ip_new = []
    # for ip in ip_list:
    #     ip_new.append(ip)
    weibo_files = os.listdir(entity_path_zh)
    weibo_files_0 = weibo_files[0:1000]
    weibo_files_1 = weibo_files[1000:2000]
    weibo_files_2 = weibo_files[2000:3000]
    weibo_files_3 = weibo_files[3000:4000]
    weibo_files_4 = weibo_files[4000:]
    # def load_cn_probase(file, ip_list):
    t0 = MyThread(cn_probase_multiprocessing,args=(weibo_files_0,ip_list))
    t1 = MyThread(cn_probase_multiprocessing, args=(weibo_files_1, ip_list))
    t2 = MyThread(cn_probase_multiprocessing, args=(weibo_files_2, ip_list))
    t3 = MyThread(cn_probase_multiprocessing, args=(weibo_files_3, ip_list))
    t4 = MyThread(cn_probase_multiprocessing, args=(weibo_files_4, ip_list))
    t0.start()
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t0.join()
    t1.join()
    t2.join()
    t3.join()
    t4.join()

    # ip_list = read_ip()
    # # ip_new = []
    # # for ip in ip_list:
    # #     ip_new.append(ip)
    # weibo_files = os.listdir(entity_path_zh)
    # weibo_files_finish = os.listdir(concept_path_zh)
    # Parallel(n_jobs=30, backend='threading')(
    #     delayed(load_cn_probase)(file, ip_list) for file in tqdm(weibo_files))
#中文isA relation获取单线程主函数
def main_isA_zh_singleprocessing():
    ip_list = read_ip()
    # ip_new = []
    # for ip in ip_list:
    #     ip_new.append(ip)
    weibo_files = os.listdir(entity_path_zh)
    cn_probase_multiprocessing(weibo_files,ip_list)
    # weibo_files_0 = weibo_files[0:1000]
    # weibo_files_1 = weibo_files[1000:2000]
    # weibo_files_2 = weibo_files[2000:3000]
    # weibo_files_3 = weibo_files[3000:4000]
    # weibo_files_4 = weibo_files[4000:]
    # # def load_cn_probase(file, ip_list):
    # t0 = MyThread(cn_probase_multiprocessing,args=(weibo_files_0,ip_list))
    # t1 = MyThread(cn_probase_multiprocessing, args=(weibo_files_1, ip_list))
    # t2 = MyThread(cn_probase_multiprocessing, args=(weibo_files_2, ip_list))
    # t3 = MyThread(cn_probase_multiprocessing, args=(weibo_files_3, ip_list))
    # t4 = MyThread(cn_probase_multiprocessing, args=(weibo_files_4, ip_list))
    # t0.start()
    # t1.start()
    # t2.start()
    # t3.start()
    # t4.start()
    # t0.join()
    # t1.join()
    # t2.join()
    # t3.join()
    # t4.join()

    # ip_list = read_ip()
    # # ip_new = []
    # # for ip in ip_list:
    # #     ip_new.append(ip)
    # weibo_files = os.listdir(entity_path_zh)
    # weibo_files_finish = os.listdir(concept_path_zh)
    # Parallel(n_jobs=30, backend='threading')(
    #     delayed(load_cn_probase)(file, ip_list) for file in tqdm(weibo_files))
def main_data_split():
    weibo_files = os.listdir(entity_path_zh)
    weibo_finish_files = os.listdir(concept_path_zh)
    weibo_unfinish_files = []
    for file in weibo_files:
        if file not in weibo_finish_files:
            weibo_unfinish_files.append(file)
    print(len(weibo_unfinish_files)/16)


def main_isA_zh_sleep():
    weibo_files = os.listdir(entity_path_zh)
    weibo_files_finish = os.listdir(concept_path_zh)
    num_count = 4990
    for j in tqdm(range(len(weibo_files))):
        file = weibo_files[j]
    # weibo_files_finish = os.listdir(concept_path_zh)
        if not file in weibo_files_finish:
            file_df = pd.read_csv(entity_path_zh+file)
            file_df_new = pd.DataFrame(columns=['mid','concept'])
            # print(file_df['entity'][0])
            # entity_isA_all = []
            for i in range(len(file_df['entity'])):
                mid = str(file_df['mid'][i]).strip('\t') + '\t'
                entity_isA_temp = []
                entity_lst = [ent.strip(' ').strip('\'') for ent in file_df['entity'][i].strip('\t').strip('[').strip(']').split(',')]
                # print(entity_lst[0])
                # cn_probase_search(entity_lst[0])

                for entity in entity_lst:
                    isA_concept = cn_probase_search_sleep(entity)
                    num_count -= 1
                    # ip_list = ip_new
                    entity_isA_temp += isA_concept
                    if num_count == 0:
                        sleep(3600)
                        num_count = 4990
                    else:
                        r_num = random.randint(1,5)
                        sleep(r_num)
                print('all:',entity_isA_temp)
                entity_isA = str(entity_isA_temp) + '\t'
                file_df_new.loc[i] = [mid, entity_isA]
            weibo_output = '../data/weibo/weibo_concept/' + file
            file_df_new.to_csv(weibo_output)

    # ip_list = read_ip()
    # # ip_new = []
    # # for ip in ip_list:
    # #     ip_new.append(ip)
    # weibo_files = os.listdir(entity_path_zh)
    # # weibo_files_finish = os.listdir(concept_path_zh)
    # Parallel(n_jobs=30, backend='threading')(
    #     delayed(load_cn_probase)(file, ip_list) for file in tqdm(weibo_files))
#读取yago知识图谱tsv数据
def read_tsv():
    isA_dict = defaultdict(list)
    csv.register_dialect('mydialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    # with open('/home/sunmengzhu2019/kg-YAGO/yagoLiteralFacts.tsv', 'r', encoding='utf-8') as csvfile:
    #     file_list = csv.reader(csvfile, 'mydialect')
    #     subject_LiteralFacts = []
    #     predicate_LiteralFacts = []
    #     object_LiteralFacts = []
    #     for line in file_list:
    #         # print(type(line))
    #         # print(len(line))
    #         # print(line[0])
    #         subject_LiteralFacts.append(line[1])
    #         predicate_LiteralFacts.append(line[2])
    #         object_LiteralFacts.append(line[3])
    # # print(subject[0:10])
    with open('/home/sunmengzhu2019/kg-YAGO/yagoTransitiveType.tsv', 'r', encoding='utf-8') as csvfile:
        file_list = csv.reader(csvfile, 'mydialect')
        subject_Facts = []
        predicate_Facts = []
        object_Facts = []
        # k = 0
        for line in file_list:
            isA_dict[line[1]].append(line[3])
            # print(type(line))
            # print(len(line)) #5 第一个和最后一个是’‘
            # print(line[1])
            # k += 1
            # if k == 3:
            #     break
            # subject_Facts.append(line[1])
            # predicate_Facts.append(line[2])
            # object_Facts.append(line[3])
    with open('/home/sunmengzhu2019/kg-YAGO/yagoSimpleTypes.tsv', 'r', encoding='utf-8') as csvfile:
        file_list = csv.reader(csvfile, 'mydialect')
        for line in file_list:
            isA_dict[line[1]].append(line[3])

    csv.unregister_dialect('mydialect')
    # subject = subject_LiteralFacts+subject_Facts+subject_DateFacts
    # predicate = predicate_LiteralFacts+predicate_Facts+predicate_DateFacts
    # object = object_LiteralFacts+object_Facts+object_DateFacts
    # subject = subject_Facts
    # predicate = predicate_Facts
    # object = object_Facts
    return isA_dict
#动态ip获取并写入txt文件
def get_ip():
    url = 'http://gec.ip3366.net/api/?key=20201115225904704&getnum=300&anonymoustype=3&area=1&proxytype=01'
    #'http://gec.ip3366.net/api/?key=20201115225904704&getnum=300&anonymoustype=3&proxytype=01'
    headers = {
        'Cookie':'Hm_lvt_c4dd741ab3585e047d56cf99ebbbe102=1621754190; Hm_lpvt_c4dd741ab3585e047d56cf99ebbbe102=1622615964',
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.193 Safari/537.36',
        'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Encoding':'gzip, deflate',
        'Accept-Landuage':'zh-CN,zh;q=0.9'
    }
    res = requests.get(url, headers=headers)
    # res = requests.get(url)
    # print(res.status_code)
    # print(res.content)
    # print(type(res.content))
    # print(res.text)
    # print(type(res.text)) #str
    with open('ip.txt','w',encoding='utf-8',newline='')as f:
        f.writelines(res.text)
#读取txt文件中的动态ip
def read_ip():
    with open('ip.txt','r',encoding='utf-8')as file:
        ip_all = file.readlines()
    ip_list = []
    for line in ip_all:
        ip = line.strip('\n').strip('\r')
        ip_list.append(ip)
    print(ip_list[0],ip_list[1])
    return ip_list


# main_isA_zh()
get_ip()
main_isA_zh_singleprocessing()

# ip_list = read_ip()
# cn_probase_search('刘德华',ip_list)
# read_ip()
# read_tsv()
# cn_YAGO_search()
# main_isA_zh_sleep()