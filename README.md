# knu-final-project
## Contributors
- [조승희](https://github.com/sa02045)
- [박세인](https://github.com/sein126)
- 이경욱
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
     - WPA(승리확률기여도)  
  
### 2.2. model 디자인 설정  
  2.2.1 케라스를 이용한 붓꽃 분류모델을 기반으로 활용.  
    
> [붓꽃분류모델](https://pinkwink.kr/1128?category=769346): 분류모델의 대표적인 예제. 꽃받침, 꽃잎의 길이, 꽃잎의 폭. 이 data들을 가지고 세 개의 종을 분류하는 모델.   
  
  2.2.2 optimizer : adam optimizer 사용  
  2.2.3 은닉층 : 4개 사용  
  
  
