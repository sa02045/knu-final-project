import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder
#데이터 불러오기
df=pd.read_csv(r"C:\Users\Lee Kyoung Wook\Desktop\asdasd\stat_data bunt_sample.csv")
#df2=pd.read_csv(r"C:\Users\Lee Kyoung Wook\Desktop\asdasd\pythonProject\stat_data bunt_sample_LG_2019.csv")
#wpa 0,1로 변경하기
df['wpa']=df['wpa'].apply(lambda x: '노우' if x<-0.01 else '번트')

for i in ['P','상황','스코어','타석','타수','득점','안타','2타','3타','홈런','루타','타점','도루','도실','볼넷','사구','고4','삼진','병살','희타','희비']:
    df=df.drop(i, axis=1)
#    df2=df2.drop(i, axis=1)

data_set=df.values

#x_predict=df2.iloc[1,:-1].values
 #print(x_predict)
x=data_set[:50,4:len(df.columns)-1].astype(float)

y=data_set[:50,-1]

e=LabelEncoder()
e.fit(y)
y=e.transform(y)

print('x:' ,str(x))
print('y":',y)
seed=0
np. random.seed(seed)
tf.random.set_seed(seed)

model=tf.keras.models.Sequential([
    tf.keras.layers.Dense(20,input_dim=13,activation='relu')
    ,tf.keras.layers.Dense(2,activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(x,y,epochs=50,batch_size=10)

print('\n Accuracy=%.4f'% (model.evaluate(x,y)[1]))
print(model.evaluate(x,y))
X=np.array([23.  ,   58. ,     0.256 ,  0.354  , 0.387 ,  0.74   , 0.336 ,100. ,     1.02,
    0.32  , -2.91  , -0.34   , 5.]).reshape(1,-1)
model_predic=model.predict(X)
print(model_predic)