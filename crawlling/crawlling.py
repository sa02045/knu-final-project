#!/usr/bin/env python
# coding: utf-8

# In[5]:


import requests
from bs4 import BeautifulSoup
import copy
import pandas as pd


# In[11]:


def get_SB_data(url):
    req = requests.get(url)
    html = req.text
    soup = BeautifulSoup(html,'html.parser')

    score_odd = soup.find_all(class_= 'oddrow_stz')
    score_even = soup.find_all(class_= 'evenrow_stz')
    sit_list = []

    for sc in score_odd:
        sit = []
#         print(sc.text, end= ' ')
#         print()
        for s in sc:
            sit.append(s.text)
        sit_list.extend([sit])   

    for sc in score_even:
        sit = []
#         print(sc.text, end= ' ')
#         print()
        for s in sc:
            sit.append(s.text)
        sit_list.extend([sit])           

    # print(sit_list)
    
    sit_list_ = copy.deepcopy(sit_list)
    new_sit_list=[]
    for sit in sit_list_:
    
#     print(sit[1])
#     print(sit[4])
#     print(sit[6])
    
        if '실패' in sit[6]: #도루 성공여부
            s = []
            s.append(sit[4]) #타자이름
            s.append(sit[1]) # 상대팀
    #         s.append(sit[6])
            s.append('실패')
            new_sit_list.append(s)
        elif '도루' in sit[6]: #도루 성공여부
            s = []
            s.append(sit[4]) #타자이름
            s.append(sit[1]) # 상대팀
    #         s.append(sit[6])
            s.append('도루')
            new_sit_list.append(s)
            
    df = pd.DataFrame([a for a in new_sit_list],
                 index = [i for i in range(len(new_sit_list))],
                 columns = ['타자','상대','도루 성공 여부'])

    return df


# In[12]:


def get_raa(url):

    req = requests.get(url)
    html = req.text
    soup = BeautifulSoup(html,"html.parser")
    score = soup.find_all('td')

    start = 0
    idx =1
    raa = []
    for sc in score:
    #     print(sc.text)
        if sc.text == '권희동' : # 요기 바꾸면 선수 아무나 크롤링가능!
            raa.append(sc.text)
            start =1
        if start > 0:
            start +=1
            if start == 6 or start ==7 or start ==8 :
                raa.append(sc.text)            
            if start == 8 : 
                break

#     print(raa)
    dk = pd.DataFrame([raa],
                     index = [1],
                     columns = ['타자','도루RAA','도루 성공','도루 실패'])

    # print(score)
    return dk
    


# In[18]:


def merge_raa_SB_data(d_r,d_s):
    c = d_r
    for i in range(len(d_s)-1):
        d_r = pd.concat([d_r,c])
#     print(dk)
    d_r = d_r.reset_index() # index 새로 설정
    d_r = d_r.drop('index',axis = 1)   
    # dk = pd.concat([db,sit_df],axis=1)
    d_s = d_s.drop(columns = ['타자'],axis=1)
    # print(df)
    da = pd.concat([d_r,d_s],axis = 1)
    # print(dk)
    return da


# In[19]:


if __name__ == '__main__':
    # url = 권희동 16년 플레이 로그 중 "주루"
    # url2 = 16년 전체 선수 도루가치
    url = 'http://www.statiz.co.kr/player.php?opt=6&sopt=0&name=권희동&re=0&da=11&year=2014&plist=&pdate='
    url2 = 'http://www.statiz.co.kr/stat.php?opt=0&sopt=0&re=0&ys=2014&ye=2014&se=0&te=&tm=&ty=0&qu=auto&po=0&as=&ae=&hi=&un=&pl=&da=11&o1=RE24_BR2&o2=WAR_ALL&de=1&lr=0&tr=&cv=&ml=1&sn=30&si=&cn='
    df = get_SB_data(url)
    dk = get_raa(url2)
    da = merge_raa_SB_data(dk,df)
    print(da)
    da.to_csv('14권희동.csv',encoding = 'euc-kr') 

