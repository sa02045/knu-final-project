#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import copy
import sys


# In[2]:


def get_stat_year(name_list,year):
    stat_url = []
    for name in name_list:
        stat_url.append('http://www.statiz.co.kr/player.php?opt=1&name='+name)

    # stat_data 가져오기
    stat_list = [[] for _ in range(len(stat_url))]
    for i in range(len(stat_url)): # 이름 넣어주기
        stat_list[i] = [name_list[i]]

    idx = 0

    for url in stat_url:

        req = requests.get(url)
        html = req.text
        soup = BeautifulSoup(html,"html.parser")
        score = soup.find_all('td')

        start = 0
        idx +=1
        #print(score)
        for sc in score:
            if sc.text == year :
                start =1
            if start > 0:
                start +=1
                if start == 33 : 
                    print()
                    break
                #print(sc.text, end=" ")
                stat_list[idx-1].append(sc.text)

    # data_head 가져오기    
    matrix = soup.find_all(class_="colhead_stz0")
    mat = []    
    for para in matrix:
        for pa in para:
            #print(pa.text, end = " ")
            if not pa.text in mat:
                mat.append(pa.text)

    mat.remove('비율')
    mat.remove('WAR*')
    mat.remove('WPA')
    mat.extend(['WAR*','WPA'])

    mat_name = ["이름"]
    mat_name.extend(mat)
    #print(mat_name)

    #print()
    #print(mat)

    #data frame 만들기
    df = pd.DataFrame(
                 columns = [i for i in mat_name])
    for i in range(len(name_list)):
        df1 = pd.DataFrame([stat_list[i]],index = [i],
                          columns = [i for i in mat_name])
        #print(df1)
        df = pd.concat([df,df1])
    
    return df
# print(stat_list)


# In[3]:


def get_BABIP(name_list,year):
    
    #babip가져오기
    #year = '2019'

    babip_url = []
    for name in name_list:
        babip_url.append('http://www.statiz.co.kr/player.php?opt=1&sopt=0&name='+ name+'&re=0&se=0&da=2')

    # print(babip_url)

    babip_list=[]
    for url in babip_url:
        req = requests.get(url)
        html = req.text
        soup = BeautifulSoup(html,"html.parser")
        score = soup.find_all('td')


        start = 0
    #     babip = 0

        for sc in score:
            if sc.text == year:
                start = 1
            if start >0 :
                start += 1
                if start == 13:
                    babip_list.append(sc.text) # bibip값 담기

#                 if start == 14:
#                     print()
                    break

#                 print(sc.text, end= " ")

    #     print(f'BABIP: {babip}')
#     print(babip_list)
    
    return babip_list


# In[4]:


def get_RE24(name_list,year):

    # RE24가져오기
    # year = '2019'
    
    RE24_url = []
    for name in name_list:
        RE24_url.append('http://www.statiz.co.kr/player.php?opt=1&sopt=0&name='+ name+'&re=0&se=0&da=4')

    # print(babip_url)

    RE24_list=[]
    for url in RE24_url:
        req = requests.get(url)
        html = req.text
        soup = BeautifulSoup(html,"html.parser")
        score = soup.find_all('td')


        start = 0
    #     babip = 0

        for sc in score:
            if sc.text == year:
                start = 1
            if start >0 :
                start += 1
                if start == 16:
                    RE24_list.append(sc.text) # RE24값 담기

#                 if start == 14:
#                     print()
                    break

#                 print(sc.text, end= " ")

    #     print(f'BABIP: {babip}')
#     print(babip_list)
    
    return RE24_list


# In[5]:


def get_bunt_situation(name , year):

    bunt_url = 'http://www.statiz.co.kr/player.php?opt=6&sopt=0&name='+name+'&re=0&da=12&year='+year+'&plist=&pdate='

    req = requests.get(bunt_url)
    html = req.text
    soup = BeautifulSoup(html,'html.parser')

    score_odd = soup.find_all(class_= 'oddrow_stz')
    score_even = soup.find_all(class_= 'evenrow_stz')
    sit_list = []

    for sc in score_odd:
        sit = []
    #     print(sc.text, end= ' ')
    #     print()
        for s in sc:
            sit.append(s.text)
        sit_list.extend([sit])   
    # print('----------------------------------------------------------------')
    for sc in score_even:
        sit = []
    #     print(sc.text, end= ' ')
    #     print()
        for s in sc:
            sit.append(s.text)
        sit_list.extend([sit])         
    # print('----------------------------------------------------------------')
    sit_list_ = copy.deepcopy(sit_list)

    s = [] 

    for sit in sit_list_:
    #     print(s)
        s_ = []
        if not ('희생플라이' in sit[6] or '실책' in sit[6]) and ('1루' in sit[7]):
            s_.append(sit[4][0]) # 타순
            s_.append(sit[2][0]) # 이닝
            s_.append(sit[7][:5]) # 상황
            s_.append(sit[7][-3:]) # 스코어
            s_.append(sit[13]) # wpa

        s.extend([s_])
    # print(s)

    while(s.count([])!= 0): #빈 행렬 제거
        s.remove([])

    df = pd.DataFrame([a for a in s],
                     index = [i for i in range(len(s))],
                     columns = ['타순','이닝','상황','스코어','wpa'])

#     print(df)
    return df


# In[5]:


def merge_data_and_exp_data(name_list, data_base, year, sit):
    df = pd.DataFrame()
        
    for name in name_list:
#         print(name)
        if sit == 'bunt':
            sit_df = get_bunt_situation(name,year)
        else : 
            sit_df = get_attk_situation(name,year)
        if len(sit_df)==0: continue
        c = data_base.loc[data_base['이름']==name]
#         print(f'c : {c}')
        db = data_base.loc[data_base['이름']==name]
#         print(f'db1: {db}')
        for i in range(len(sit_df)-1):
            db = pd.concat([db,c])
#         print(f'db2: {db}')    
        db = db.reset_index() # index 새로 설정
        db = db.drop('index',axis = 1)   
        db = pd.concat([db,sit_df],axis=1)
#         print(db)
        df = pd.concat([df,db]) # 하나의 data_set으로 모두 합쳐줌
#     print(df)
    return df

