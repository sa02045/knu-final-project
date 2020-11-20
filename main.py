import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import juru_summary
'''
#데이터 불러오기
df=pd.read_csv(r"juru_data.csv")
# print(df.head())

df = df.dropna() # 행기준 nan 있는 값 제거
for i in ['상대','타자','도루','도실','도루기회','도루시도%']:
    df=df.drop(i, axis=1)
'''
'''
print(df.columns) # 남은 columns 확인 
# Index(['도루RAA', '도루 성공', '도루 실패', '도실%', '도루 성공 여부'], dtype='object')
'''
'''
# print(f'df : \n {df.head(10)}')

############# drop_duplicate #####################
df_d_d = df.drop_duplicates(['도루RAA','도루 성공','도루 실패','도실%'],keep ='first')
# print(f'df_after_drop_duplicate : \n {df_d_d.head(10)}')
# print(f'df_d_d.shape() : {df_d_d.shape}') # shape : (94,5)
############# drop_duplicate #####################

data_set = df_d_d.values
# print(len(data_set)) # 94 dataframe shape와 동일
'''
# history
def show_graph(history):
    history_dict = history.history
    accuracy = history_dict['accuracy']
    val_accuracy = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(16, 1))

    plt.subplot(121)
    plt.subplots_adjust(top=2)
    plt.plot(epochs, accuracy, 'ro', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy and loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy and Loss')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
               fancybox=True, shadow=True, ncol=5)
    plt.legend(bbox_to_anchor=(1, -0.1))

    plt.subplot(122)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
               fancybox=True, shadow=True, ncol=5)
    plt.legend(bbox_to_anchor=(1, 0))

    plt.show()

years = ['2014','2015','2016','2017','2018','2019','2020']
df = juru_summary.juru_summary(years)
data_set = df.values

x=data_set[:,:len(df.columns)-1].astype(float)
print(len(x)) # 934
y=data_set[:,-1].astype(str)
# print(f'y_original : {y}')

######### y data encoding #############
e=LabelEncoder()
e.fit(y)
y_encoded = e.transform(y)
# print(f'y_encoded1 : {y_encoded}')

# y_encoded[y_encoded==1]=0
# print(f'y_encoded3 : {y_encoded}')
y_encoded[y_encoded==2]=0
# print(f'y_encoded2 : {y_encoded}') # 성공 : 1, 실패 : 0 로 encoding
############y data encoding ################


seed=7
np. random.seed(seed)
tf.random.set_seed(seed)
train_indices = np.random.choice(len(x[:-50]), round(len(x[:-50])*0.8), replace=False)
val_indices = np.array(list(set(range(len(x[:-50]))) - set(train_indices)))

x_train = x[train_indices]
x_val = x[val_indices]
y_train = y_encoded[train_indices]
y_val = y_encoded[val_indices]

# # training set
# x_train = x[:len(x)-100]
# y_train = y_encoded[:len(y_encoded)-100]
# test set
x_test = x[-50:]
y_test = y_encoded[-50:]
# validation set
# x_val = x[-100:-10]
# y_val = y_encoded[-100:-10]

# from sklearn.model_selection import KFold
# num_folds = 4
# kfold = KFold(n_splits=num_folds, shuffle=True)
# fold_no = 1
# for train, test in kfold.split(x,y_encoded):
# modeling
model=tf.keras.models.Sequential([
    tf.keras.layers.Dense(16,input_dim=4,activation='relu', kernel_initializer = 'normal'),
    tf.keras.layers.Dense(16,activation='relu', kernel_initializer = 'normal'),
    tf.keras.layers.Dense(16,activation='relu', kernel_initializer = 'normal'),
    tf.keras.layers.Dense(16,activation='relu', kernel_initializer = 'normal'),
    tf.keras.layers.Dense(16,activation='relu', kernel_initializer = 'normal'),
    tf.keras.layers.Dense(1,activation='sigmoid', kernel_initializer = 'normal')
])

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'], )

history = model.fit(x_train,y_train,epochs=50, batch_size = 30, validation_data=(x_val,y_val))
# history = model.fit(x[train],y_encoded[train],epochs=250, batch_size = 30)
# print(f'model.evaluate(x,y) : {model.evaluate(x[test],y_encoded[test])}')
# fold_no = fold_no + 1
# print('\n Accuracy=%.4f'% (model.evaluate(x,y_encoded)[1]))
print(f'model.evaluate(x,y) : {model.evaluate(x_train,y_train)}')


model_predic=model.predict(x_test)
print(f'model_predic of x: {model_predic}')
print(f'y_encoded[-10:] : {y_encoded[-50:]}')

show_graph(history)