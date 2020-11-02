#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import copy
import sys
import import_ipynb
import Data_Functions


# In[2]:


if __name__ == '__main__':
#     name_list = ['박해민','구자욱','강민호','러프','이학주','김헌곤','김동엽','박계범']
    name_team_list = {
                    '삼성' : ['박해민','구자욱','이학주','김헌곤','김동엽','박계범']
                   }
    year = '2019'
    for team in name_team_list:
        data_base = Data_Functions.get_stat_year(name_team_list[team],year)
        BABIP = Data_Functions.get_BABIP(name_team_list[team],year)
        RE24 = Data_Functions.get_RE24(name_team_list[team], year)

        #print(data_base)
        data_base.insert(31,'BABIP',BABIP)
        data_base.insert(32,'RE24',RE24)
        print(data_base)

        db_sample = Data_Functions.merge_data_and_exp_data(name_team_list[team], data_base, year,'bunt') # merge된 dataset을 db_sample에 리턴
#         db_sample.to_csv('stat_data/bunt_sample_'+team+'_'+year+'.csv',encoding = 'euc-kr') # bunt_sample.csv 파일로 저장
        print(db_sample)

