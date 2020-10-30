# knu-final-project
## Contributors
- [조승희](https://github.com/sa02045)
- [박세인](https://github.com/sein126)
- 이경욱
- 임동영
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
* Statiz 야구데이터사이트 http://www.statiz.co.kr/main.php 사이트에서 웹 크롤러 개발 후 데이터 모으기
<hr>

2주차
=====
## 1. 기계학습 모델 디자인

### 1.1. About Model 
  1.1.1 간소화 -> 특정한 상황(주자 1루 상황)에서 번트 시행 여부를 결정
  
      1) 사용할 모델 : Regression Vs Classfication Vs Forest    
      - Classfication
        - 투수가 던질 공의 위치와 구질을 학습시킨 뒤 다음에 투수가 던질 구종과 위치를 예측하여 작전을 판단하자
      - Random Forest
        - 입력 데이터를 적절한 상황에 맞게 분류만 해주고 사람이 판단을 하자 
      - Regression
        - 상황과 타자의 성적, 작전 종류, 기대 득점, 득점 확률을 넣어 학습 시킨 뒤 특정 상황에서 득점 확률, 기대 득점을 예측하여 작전을 판단하자
        
        
      2) Dataset









