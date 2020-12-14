# knu-final-project
## Contributors
- [조승희](https://github.com/sa02045)
- [박세인](https://github.com/sein126)
- [이경욱](https://github.com/dlruddnr)
- [임동영](https://github.com/imdognyoung)
<hr>    

1주차
======
## 1. 개발환경 세팅

### 1.1. Github 레포지터리 만들기   
  1.1.1. 팀원 contributor 추가   
  1.1.2. git 사용법 공유   
  1.1.3. 수집된 데이터 공유   
  1.1.4. 소스코드 관리   

### 1.2. 개발 환경 통일

   1.2.1. IDE = pycharm community (anaconda 연동)   
   1.2.2. 팀원들 노트북 OS이 윈도우이기 때문에 gitbash 사용 , 교육자료 공유   
   1.2.3. gitbash - pycharm 연동   
   1.2.4. Anaconda Virtual Env 가상환경   
   1.2.5. 개발 패키지 버전 동일하게 설정   
   
      >python=3.7    
      >pandas=1.1.3    
      >tensorflow=2.1.0    
      >matlib=3.2    
        * 계속 추가 예정

### 2. github test 커밋 , 사용법 공유

### 다음주 계획
* [Statiz 야구데이터사이트](http://www.statiz.co.kr/main.php)에서 웹 크롤러 개발 후 데이터 모으기
<hr>

2주차
=====
## 1. 기계학습 모델 디자인(2020.10.30)

### 1.1. About Model 
  1.1.1 간소화 -> 특정한 상황(주자 1루 상황)에서 번트 시행 여부를 결정
  
      1) 사용할 모델 선정 : Regression Vs Classfication Vs Forest    
      - Classfication
        - 투수가 던질 공의 위치와 구질을 학습시킨 뒤 다음에 투수가 던질 구종과 위치를 예측하여 작전을 판단하자
      - Random Forest
        - 입력 데이터를 적절한 상황에 맞게 분류만 해주고 사람이 판단을 하자 
      - Regression
        - 상황과 타자의 성적, 작전 종류, 기대 득점, 득점 확률을 넣어 학습 시킨 뒤 특정 상황에서 득점 확률, 기대 득점을 예측하여 작전을 판단하자
                
### 1.2. Data set   
   1.2.1 Data set    
      - statize 에서 크롤링해와서 사용    
      - 입력으로 이닝, 스코어, 타자의 세부성적(비율 스탯보다는 wRC+같은 세이버 매트릭스)과 번트를 칠 것인지 안칠것인지를 O,X로 표현하여 학습시킨다.    
      - O,X 여부는 번트 후 WPA가 임계값보다 높으면 승리에 기여한 번트라고 판단, O라고 학습시킨다. WPA가 임계값 보다 낮으면 X라고 학습    
- 예시

X | Y 
----|----
8 말/6:8/wOBA:0.339/BA
BIP:0.347/+wRC:86.5 | WPA : 0.008 -> O
4 초/0:0/wOBA:0.339/BABIP:0.347/+wRC:86.5 | WPA : -0.021 -> x


> [wRC+](https://namu.wiki/w/wRC+): 조정득점생산력

> [WPA](https://namu.wiki/w/%EC%8A%B9%EB%A6%AC%20%ED%99%95%EB%A5%A0%20%EA%B8%B0%EC%97%AC%EB%8F%84): 승리확률기여도
      
## 2. model 확정

### 2.1. Data 조건 설정
  2.1.1. input으로 넣을 data는 어떻게 정의할 것인가?  
     - 이닝, 스코어, 타자의 세부성적(타율 / 출루율 / 장타율 / wOBA(가중출루율) / wRC+(조정득점생산력) 등 아직 정해지지 않음(수요일에 결정할 예정))  
       
  2.1.2. ouput으로는 어떤 data를 출력할 것인가?  
     - O/X (예측된 WPA값이 임계값인 -0.01 이상인 경우 O, 그렇지 않으면 X 출력)  
  
### 2.2. model 디자인 설정  
  2.2.1 케라스를 이용한 붓꽃 분류모델을 기반으로 활용.  
    
> [붓꽃분류모델](https://pinkwink.kr/1128?category=769346): 분류모델의 대표적인 예제. 꽃받침, 꽃잎의 길이, 꽃잎의 폭. 이 data들을 가지고 세 개의 종을 분류하는 모델.   
  
  2.2.2 optimizer: adam optimizer 사용  
  2.2.3 은닉층: 4개 사용  
  2.2.4 활성함수: relu
  
  2.2.5 결과: 입력값에 대한 학습 정확도 70%, loss 0.89, predict를 이용하여 출력값(O/X)의 확률이 표시됨.
  
  ### 다음주 계획 
   - 교수님께서 말씀하신 도루 작전에 대해 입력값과 출력값에 대해 결정하기. 

<hr>

3주차
=====
   
## 1. 도루전략 데이터 수집 및 모델링 

### 1.1. 데이터 수집 
  
  3.1.1 타자 데이터: 도루RAA,	도루 성공,	도루 실패, 성공 여부   
  3.1.2 포수 데이터: 도루,	도실,	도실%	,도루기회,	도루시도%,	도루   
  3.1.3 수집한 데이터 중 유의미한 데이터 선정 : 도루RAA, 도루성공, 도루실패, 도실%, 성공여부


### 1.2. 딥러닝 모델링 및 결과 확인
     
   1.2.1 모델링 설정
   
      1) 사용할 모델 선정 : keras.models.Sequential() 사용   
      -input_노드 : 4개
      -output_노드 : 2개 (0:성공, 1:살패)
      -히든층: 1개
      -히든노드 : 15개
      -epoch_size : 50
      -batch_size : 10
      -활성화 함수 : relu(input-> 히든층), sigmoid(히든층-> output)
      
   1.2.2 결과확인
   
   
      -결과: epoch_size를 증가시키면 loss값 감소 확인
      모델의 정확도는 1.00 loss 값은 약 0.004
      임의의 값을 넣어 모델 확인시, 데이터: [1.38,14,3,7.8]
      [0.91,0.05]로 도루시 성공이 높은 것을 확인
      실제 경기결과와 일치


<hr>

4주차
=====

## 1. 도루 작전 모델링 수정 및 보완   

  ### 1.1. 데이터 양 증가 및 중복 데이터 삭제   
  
  1.1.1. 도루 작전에 대한 데이터 프레임 확정   
  X | Y 
----|----
도루RAA, 타자의 도루시도, 타자의 도루 성공, 포수의 도루 저지율 | 성공 -> 1
도루RAA, 타자의 도루시도, 타자의 도루 성공, 포수의 도루 저지율 | 실패 -> 0
  
  1.1.2. 데이터 양 증가   
    - 2014~2020 까지 약 930여개의 데이터 축적.   
    (단, 포수의 도루저지율 데이터는 구단별로 해당 연도 출장 수가 가장 많은 포수의 도루 저지율로 통일)   
    
  1.1.3. 데이터 중복 삭제    
    - 데이터가 중복되는 경우나, nan 데이터의 경우는 삭제    
    
```python
df = df.dropna()    
df_d_d = df.drop_duplicates(['도루RAA', '도루 성공', '도루 실패', '도실%'], keep='first')
```

  ### 1.2. 모델 수정
  
  1.2.1 모델 수정 전 문제점
  - 저번 주의 모델에서 치명적인 문제점 발견
    1) 데이터 셋을 50개 정도만 사용
    -> 그마저도 매우 중복된 데이터
    2) y의 경우 encoding하는 중에 데이터셋의 모든 y가 0으로 바뀌어버리는 현상 발생
    3) model이 binary_crossentropy를 사용하는 이진 분류 모델이 아닌 sparse categorical crossentropy를 사용하는 다중 분류 함수 모델로 구성
    -> 다중 분류 함수 모델로 구성하였으나 출력단에서 활성화 함수로 sigmoid 함수 사용
    4) training set, validation set, test set 구분을 하지않은 채로 model training만 진행
    -> 정량적 평가가 불가능
    
  1.2.2 모델 수정
```python
def create_model(optimizer = 'rmsprop', init ='glorot_uniform'):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16,input_dim=4,activation='relu', kernel_initializer = init),
        tf.keras.layers.Dense(16,activation='relu', kernel_initializer = init),
        tf.keras.layers.Dense(16,activation='relu', kernel_initializer = init),
        tf.keras.layers.Dense(16,activation='relu', kernel_initializer = init),
        tf.keras.layers.Dense(16,activation='relu', kernel_initializer = init),
        tf.keras.layers.Dense(1,activation='sigmoid', kernel_initializer = init)
    ])
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer , metrics=['accuracy'])
    return model

# 6개의 hidden layer 구성
# 각 layer마다 output node는 16개로 구성 및 활성화 함수는 'relu' 사용
# deep 하지만, gradient vanishing이 일어나지 않도록 narrow하게 구성(model을 wide하고 short하게 짜는 것 보다 이 방법이 나았음)
# optimizer는 adam 보다 RMSprop(Root Mean Square propagation)사용
# rmsprop => 경사 하강법을 빠르게 하는 모멘텀과 같은 알고리즘 중 하나.손실함수의 최저점으로 나아갈 때,
#            불필요하게 수직 방향으로 진동하는 것을 제한함으로써 빠르게 학습이 이루어지게 유도함.
# 
# 이진 분류 이므로 출력단 node = 1, 활성화 함수는 sigmoid 사용
# kerner_initializer로 xavier 방식을 통한 weight initialization 실행
```   

```python
model.summary()
```
  Layer (type)         |        Output Shape      |        Param #   
  -----|-------------------------------|---------------
dense (Dense)          |      (None, 16)           |     80        
dense_1 (Dense)       |       (None, 16)          |      272       
dense_2 (Dense)        |      (None, 16)          |      272       
dense_3 (Dense)        |      (None, 16)           |     272       
dense_4 (Dense)         |     (None, 16)         |       272       
dense_5 (Dense)        |      (None, 1)           |      17        

Total params: 1,185   
Trainable params: 1,185   
Non-trainable params: 0   

  1.2.3 모델 보완
    * dataset을 training set, validation set, test set으로 나누어 검증 실행
    -> K-fold cross-validation을 통한 검증
    
  ## 2. model 평가
    
  ### 2.1 random seed를 통한 training set, validation set 분류에 의한 평가
  2.1.1 set 분리
  ```python
  seed=7
  np. random.seed(seed)
  tf.random.set_seed(seed)
  train_indices = np.random.choice(len(x[:-50]), round(len(x[:-50])*0.8), replace=False)
  val_indices = np.array(list(set(range(len(x[:-50]))) - set(train_indices)))
  .
  .
  .
  x_test = x[-50:]
  y_test = y_encoded[-50:]
  ```
  epoch = 200, batchsize = 80 으로 실행
  ```
  model.evaluate(x,y) : [0.5582096909701065, 0.7340877]
  ```   
  => 다양한 epoch와 batchsize로 모델을 training 시켰으나, 결국 test set 고정으로 인해 어떤 값이 더 적합한 하이퍼 파라미터인지 애매   
  => *K-fold cross-validation*을 통해 검증
  
  ### 2.2 K-fold cross-validation
  2.2.1 다양한 경우의 하이퍼 파라미터 셋 결정
  ```python
  kfold = StratifiedKFold(n_splits = 5, shuffle=True, random_state = seed)
  ```
  => fold 수 : 5   
  ```python
  optimizers = ['rmsprop', 'adam']
  inits = ['glorot_uniform', 'normal']
  epochs = [50, 100, 150]
  batches = [30, 50]
  ```
  -> 이중에서 best evaluation을 결정   
  
  ```linux
  Best: 0.729124 using {'batch_size': 30, 'epochs': 50, 'init': 'normal', 'optimizer': 'rmsprop'}
  0.728049 (0.002355) with: {'batch_size': 30, 'epochs': 50, 'init': 'glorot_uniform', 'optimizer': 'rmsprop'}
  0.726985 (0.002949) with: {'batch_size': 30, 'epochs': 50, 'init': 'glorot_uniform', 'optimizer': 'adam'}
  0.729124 (0.002313) with: {'batch_size': 30, 'epochs': 50, 'init': 'normal', 'optimizer': 'rmsprop'}
  0.729124 (0.002313) with: {'batch_size': 30, 'epochs': 50, 'init': 'normal', 'optimizer': 'adam'}
  0.725916 (0.004909) with: {'batch_size': 30, 'epochs': 100, 'init': 'glorot_uniform', 'optimizer': 'rmsprop'}
  0.725916 (0.004909) with: {'batch_size': 30, 'epochs': 100, 'init': 'glorot_uniform', 'optimizer': 'adam'}
  0.729124 (0.002313) with: {'batch_size': 30, 'epochs': 100, 'init': 'normal', 'optimizer': 'rmsprop'}
  0.729124 (0.002313) with: {'batch_size': 30, 'epochs': 100, 'init': 'normal', 'optimizer': 'adam'}
  0.725910 (0.005212) with: {'batch_size': 30, 'epochs': 150, 'init': 'glorot_uniform', 'optimizer': 'rmsprop'}
  0.725904 (0.002737) with: {'batch_size': 30, 'epochs': 150, 'init': 'glorot_uniform', 'optimizer': 'adam'}
  0.729124 (0.002313) with: {'batch_size': 30, 'epochs': 150, 'init': 'normal', 'optimizer': 'rmsprop'}
  0.729124 (0.002313) with: {'batch_size': 30, 'epochs': 150, 'init': 'normal', 'optimizer': 'adam'}
  0.726985 (0.002949) with: {'batch_size': 50, 'epochs': 50, 'init': 'glorot_uniform', 'optimizer': 'rmsprop'}
  0.729124 (0.002313) with: {'batch_size': 50, 'epochs': 50, 'init': 'glorot_uniform', 'optimizer': 'adam'}
  0.729124 (0.002313) with: {'batch_size': 50, 'epochs': 50, 'init': 'normal', 'optimizer': 'rmsprop'}
  0.729124 (0.002313) with: {'batch_size': 50, 'epochs': 50, 'init': 'normal', 'optimizer': 'adam'}
  0.725910 (0.002072) with: {'batch_size': 50, 'epochs': 100, 'init': 'glorot_uniform', 'optimizer': 'rmsprop'}
  0.725916 (0.004909) with: {'batch_size': 50, 'epochs': 100, 'init': 'glorot_uniform', 'optimizer': 'adam'}
  0.729124 (0.002313) with: {'batch_size': 50, 'epochs': 100, 'init': 'normal', 'optimizer': 'rmsprop'}
  0.729124 (0.002313) with: {'batch_size': 50, 'epochs': 100, 'init': 'normal', 'optimizer': 'adam'}
  0.725904 (0.002737) with: {'batch_size': 50, 'epochs': 150, 'init': 'glorot_uniform', 'optimizer': 'rmsprop'}
  0.724840 (0.002440) with: {'batch_size': 50, 'epochs': 150, 'init': 'glorot_uniform', 'optimizer': 'adam'}
  0.729124 (0.002313) with: {'batch_size': 50, 'epochs': 150, 'init': 'normal', 'optimizer': 'rmsprop'}
  0.729124 (0.002313) with: {'batch_size': 50, 'epochs': 150, 'init': 'normal', 'optimizer': 'adam'}
  ```
  => 검증 결과 {'batch_size': 30, 'epochs': 50, 'init': 'normal', 'optimizer': 'rmsprop'} 일때 best evaluation 발생   
  > 모델 weight 저장 : https://blog.naver.com/asdjklfgh97/222109500203    
  > 콜백함수 만들기 : https://www.tensorflow.org/guide/keras/train_and_evaluate    
  > k-fold cross-validation : https://3months.tistory.com/321   
  > 베스트 하이퍼 파라미터 선정하기 : https://blog.naver.com/trimurti/221379458846    
  
  ## 3. 고찰    
  ### 3.1 model depth, wide
  * depth는 너무 깊으면 gradient vanishing이 일어날 가능성이 높다.
  * wide하게 model을 구성한다면 parameter가 지나치게 많아진다. input으로 들어가는 x_parameter가 고작 4개 뿐인데 대규모의 parameter가 필요할까?
  * 결론적으로 x를 구분해줄 x_parameter가 너무 적다는 생각이 들었다.
  
  ### 3.2 optimizer
  * rmsprop와 adam에서 accuracy차이는 그닥 없다. 다만 rmsprop가 특정 환경에서 더 빠른 모멘텀을 가지고 경사하강을 실행하기 때문에 epoch가 적은 상황에서는 adam보다 accuracy가 더 좋은 모습을 보여주는듯 하다.
  
  ### 3.3 epoch와 batch_size
  * batch_size는 작을 수록 미세하게나마 성능이 향상된다.
  * epoch가 증가할 수록 accuracy가 증가하는 경향이 매우 적다. 즉 epoch를 많이 해봤자 큰 의미가 없다.

  ### 3.4 initialization
  * initializer로 xavier 방식의 'glorot_uniform'(입력값과 출력값 사이의 난수를 선택해서 입력값의 제곱근으로 나눈다)와 정규화 된 수를 생성해주는 'normal' 방식 중 하나를 선택했다.
  * 'normal' 방식이 'glorot_uniform' 방식보다 평균적으로 0.3% 가량 높은 accuraccy를 보여줬다.
  * 이는 유의미한 수치로, 노드의 입출력값 사이에 어떠한 의미가 있을 것이라 생각한다
  
  ## 4. 총평
  * 전체적으로 결과가 많이 아쉽다. -> 생각한 만큼 accuracy가 나오지 않았다.
  * 데이터 set 갯수를 늘리는 것도 한계가 있다. -> statize에서 더 이상 긁어오는 것도 힘들다
  * test score와 k-fole cross-validation score가 좀 다르다. test score가 좀 더 높게 나온다. 
  * test set을 통한 예측 시, 예측 값의 bias가 한쪽으로 몰린 느낌이다. 이를 수정 보완 할 필요가 있어 보인다.
  * x_parameter를 늘리던가, batch_normalization 과 같은 기법으로 데이터들간의 correlation 과 같은 '관계'들을 유지시켜 줄 필요가 있다고 생각한다.
  
  <hr>

5주차
=====

## 1. 데이터의 x parameter 추가
1.1.1 *SPD*와 *RAA 주루* column 추가
```
SPD : 해당 선수가 주자로 있을 때의 효용 가치
RAA 주루 : 평균 대비 도루 득점 기여
```   
X | Y 
----|----
도루RAA, 타자의 도루시도, 타자의 도루 성공, 포수의 도루 저지율, SPD, RAA 주루 | 1

## 2. 모델 수정

2.1.1 모델 layer 변화
```python
def create_model2(optimizer = 'rmsprop', init ='normal'):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128,input_dim=6,activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer = init),
        tf.keras.layers.Dense(128,activation=tf.keras.layers.LeakyReLU(alpha=0.1)
, kernel_initializer = init),
        tf.keras.layers.Dense(128,activation=tf.keras.layers.LeakyReLU(alpha=0.1)
, kernel_initializer = init),
        tf.keras.layers.Dense(64,activation=tf.keras.layers.LeakyReLU(alpha=0.1)
, kernel_initializer = init),
        tf.keras.layers.Dense(1,activation='sigmoid', kernel_initializer = init)
    ])
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer , metrics=['accuracy'])
    return model
# 5층의 layer 사용
# 각 layer의 output parameter수를 늘려서 wide하게 구성
# 활성화 함수를 leaky relu로 변경
```

```python
model.summary()
```

  Layer (type)         |        Output Shape      |        
  -----|-------------------------------|
dense (Dense)          |      (None, 128)           |           
dense_1 (Dense)       |       (None, 128)          |           
dense_2 (Dense)        |      (None, 128)          |     
dense_3 (Dense)        |      (None, 64)           |     
dense_4 (Dense)         |     (None, 1)         |       
    
* 5층의 layer 사용하고 각 layer의 output parameter수를 늘려서 wide하게 구성    
=> dataset의 x_parameter가 증가되었기때문에 parameter 수를 증가
*  활성화 함수를 leaky relu로 변경   
=> 데이터셋에 주루RAA값이 추가 되면서 음수 영역이 생성되었는데, 이 값을 보존시키기 위해 leaky relu로 변경   
=> leakyrelu의 alpha parameter가 0.1로 설정한 것을 고려, 네트워크의 depth가 깊어지면 gradient vanishing이 일어날 수도 있을 것이라 생각해 depth를 1단계 줄였다.    

2.2.2 모델 평가
  * 아래는 개선된 모델에 대해 relu함수와 leakyrelu함수를 사용한 K-fold cross-validation 결과이다.(k = 5)
  
```
Relu : Accuracy: 71.83%  0.85%
------------------------------------------------------------------------
Leaky relu : Accuracy: 72.68%  0.18%
```
=> leaky relu 함수를 사용한 모델이 기존의 relu함수를 사용한 모델보다 accuracy가 1프로 정도 더 높게 나왔다.

  * 최적의 hyper-parameter를 찾기위해 저번과 같이 sckit learn의 GridSearchCV를 사용하여 다양한 경우의 accuracy를 비교하였다.
```
Best: 0.726795 using {'batch_size': 30, 'epochs': 50, 'init': 'normal', 'optimizer': 'rmsprop'}
0.726795 (0.001773) with: {'batch_size': 30, 'epochs': 50, 'init': 'normal', 'optimizer': 'rmsprop'}
0.724673 (0.005675) with: {'batch_size': 30, 'epochs': 50, 'init': 'normal', 'optimizer': 'adam'}
0.721520 (0.001769) with: {'batch_size': 30, 'epochs': 50, 'init': 'glorot_uniform', 'optimizer': 'rmsprop'}
0.724684 (0.001928) with: {'batch_size': 30, 'epochs': 50, 'init': 'glorot_uniform', 'optimizer': 'adam'}
0.705731 (0.022234) with: {'batch_size': 30, 'epochs': 100, 'init': 'normal', 'optimizer': 'rmsprop'}
0.722579 (0.003818) with: {'batch_size': 30, 'epochs': 100, 'init': 'normal', 'optimizer': 'adam'}
0.703598 (0.018785) with: {'batch_size': 30, 'epochs': 100, 'init': 'glorot_uniform', 'optimizer': 'rmsprop'}
0.703598 (0.013216) with: {'batch_size': 30, 'epochs': 100, 'init': 'glorot_uniform', 'optimizer': 'adam'}
0.725742 (0.002930) with: {'batch_size': 50, 'epochs': 50, 'init': 'normal', 'optimizer': 'rmsprop'}
0.722568 (0.004567) with: {'batch_size': 50, 'epochs': 50, 'init': 'normal', 'optimizer': 'adam'}
0.717304 (0.013461) with: {'batch_size': 50, 'epochs': 50, 'init': 'glorot_uniform', 'optimizer': 'rmsprop'}
0.724690 (0.003436) with: {'batch_size': 50, 'epochs': 50, 'init': 'glorot_uniform', 'optimizer': 'adam'}
0.725742 (0.002930) with: {'batch_size': 50, 'epochs': 100, 'init': 'normal', 'optimizer': 'rmsprop'}
0.724690 (0.003436) with: {'batch_size': 50, 'epochs': 100, 'init': 'normal', 'optimizer': 'adam'}
0.663559 (0.051594) with: {'batch_size': 50, 'epochs': 100, 'init': 'glorot_uniform', 'optimizer': 'rmsprop'}
0.718340 (0.007669) with: {'batch_size': 50, 'epochs': 100, 'init': 'glorot_uniform', 'optimizer': 'adam'}
```
=> best model로 'batch_size': 30, 'epochs': 50, 'init': 'normal', 'optimizer': 'rmsprop', 즉 저번과 같은 hyper-parameter가 선택되었다.

## 3. 고찰

3.1.1. 모델
* leaky relu : 데이터셋에 주루RAA값이 추가 되면서 음수 영역이 생성되었는데, 이 값을 보존시키기 위해 leaky relu로 변경 => 당연히 leaky relu를 사용해야한다고 생각한다. 다만 음수쪽 기울기인 alpha parameter = 0.1 인데 이것은 차후 수정해줄 필요가 있다고 생각한다.
* 모델의 depth와 wide : 네트워크의 depth가 깊어지면 gradient vanishing이 일어날 수도 있을 것이라 생각해 depth를 1단계 줄였다. 하지만 input parameter의 수가 늘어났기 때문에 wide하게 구성하였다. => 대신 parameter수가 너무 늘어났다.

3.1.2 성능
* 성능향상이 별로 없는 이유?
=> Data set의 분포를 살펴봐야한다고 생각한다. 여전히 우리가 만들어낸 model의 output은 0 ~ 0.3 범위에 몰려있다. 이것은 bias가 잘못된 것이라 할 수 있는데, 이는 data set안의 자료의 분포에서 답을 찾을 수 있다고 생각한다. 우선 주루RAA, 도루RAA, 그리고 포수의 도루 저지율의 상관관계에 대해 생각해 봐야한다. Sequential한 model의 특성상 layer의 진행방향에 따라 weight가 곱해지는 형태라고 할 수 있다. 이 때, RAA는 절댓값이 커봐야 7이하인 반면, 도루저지율 같은 경우는 퍼센테이지로 30 ~ 40 범위에 매우 빼곡하게 밀집되어있다. 이에 RAA와 도루저지율의 분산은 비슷할 수 있으나, median한 값이 달라서 correlation이 layer을 지나면서 계속 변화하는 것으로 보여진다. 이를 위해 layer마다 batch normalization 기법을 적용하여 layer가 지날수록 각 parameter간의 상관관계가 변하지 않도록 하는 방안도 생각되어야 한다. 그렇지 않으면 애초에 데이터 셋을 전부 normalize하게 처리 해주고 모델에 적용시키는 것도 한 방법이라고 생각이 된다.

<hr>

6주차
=====
기말고사 준비로 인해 쉬어갑니다.   
<hr>

7주차
=====
기말고사 준비로 인해 쉬어갑니다.
<hr>

