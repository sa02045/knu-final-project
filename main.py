#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import copy
import sys
<<<<<<< HEAD
# import import_ipynb
=======
>>>>>>> cd6c0f5cf7fc48931e286bad228a45b4b29a35bb
import Data_Functions


# In[2]:


if __name__ == '__main__':
#     name_list = ['박해민','구자욱','강민호','러프','이학주','김헌곤','김동엽','박계범']
    name_team_list = {
                    'all_team' : ['박해민','구자욱','강민호','러프','이학주','김헌곤','김동엽','박계범',
                            '정수빈', '최주환','김재환','허경민','박세혁','오재일','김재호',
                            '박민우', '나성범', '양의지', '박석민','권희동','노진혁','김성욱',
                            '이정후','김하성','서건창','박동원','김혜성',
                            '최정','로맥','최항','김강민','한동민','채태인',
                            '송광민','정근우','하주석','정은원','최진행',
                            '나지완','최형우','김선빈','안치홍','김주찬',
                            '이대호', '전준우','손아섭','정훈','신본기',
                            '박용택','오지환','김민성','이형종','유강남',
                            '황재균','장성우','유한준','강백호','박경수','로하스'
                            ]
                   }
    year = '2019'
    for team in name_team_list:
        data_base = Data_Functions.get_stat_year(name_team_list[team],year)
        BABIP = Data_Functions.get_BABIP(name_team_list[team],year)
        RE24 = Data_Functions.get_RE24(name_team_list[team], year)

        #print(data_base)
        data_base.insert(31,'BABIP',BABIP)
        data_base.insert(32,'RE24',RE24)
        # print(data_base)

        db_sample = Data_Functions.merge_data_and_exp_data(name_team_list[team], data_base, year,'bunt') # merge된 dataset을 db_sample에 리턴
        db_sample.to_csv('data/bunt_sample_'+team+'_'+year+'.csv',encoding = 'euc-kr') # bunt_sample.csv 파일로 저장
        # print(db_sample)

