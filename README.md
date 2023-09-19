![image](https://github.com/rimgosu/DeepLearning/assets/120752098/e0b6d433-cecb-4397-a2c1-5dbfe2e820d2)


# DeepLearning



## Text Mining

![image](https://github.com/rimgosu/DeepLearning/assets/120752098/1d5fb433-4e4f-4277-b45f-9b402f87611f)

### 9월 13일 (머신러닝 복습, 텍스트 마이닝)
> ex01. TextMining(영화리뷰 감성분석)
#### 머신러닝 종류
1. 지도학습
   - 분류(ㅂ) : 정답 데이터가 범주형(ㅂ), <br> class : 정답 데이터의 갯수 <br> ※로지스틱 회귀 : 분류모델(얜 특이하게회귀지만 분류모델임)
   - 회귀(ㅇ) : 정답 데이터가 연속형(ㅇ)
2. 비지도학습
3. 강화학습

#### 머신러닝 분석 프로세스
1. 문제정의
2. 데이터 수집
3. 데이터 전처리
4. EDA (탐색적 데이터 분석) - 시각화
5. 모델 선택 및 하이퍼파라미터 선택
6. 모델 학습 (fit)
7. 모델 예측 및 평가 (predict, score)

#### 텍스트마이닝
- 자연어 처리 방식 : Natural Language Processing, NLP
- 자연어 : 일상생활에 사용되는 언어
- 인공언어 : python, 수학식 등

#### 텍스트마이닝의 영역
1. 텍스트 분류
2. 감성 분석 (*)
3. 텍스트 요약
4. 텍스트 군집화 및 유사도 분석

#### 텍스트 데이터의 구조
> 말뭉치 >> 문서 >> 문단 >> 문장 >> 다어 >> 형태소
* 형태소 : 일정한 의미가 있는 가장 작은 말의 단위 (ex: 첫/사랑)

#### 텍스트 마이닝 분석 프로세스
1. 텍스트데이터 수집(크롤링)
2. 텍스트 전처리
3. 토큰화(벡터로 변환, 쪼개기 - 형태소 분석기) <br> a. 단어단위 <br> b. 문자단위 <br> c. n-gram 단위(n개의 연속된 단어를 하나로 취급) <br> ex) ICE/CREAM => ICECREAM <br><br>
4. 특징 값 추출(중요한 단어 추출) <br> TF-IDF : "문서"내에서 중요한 단어여야함. <br>"모든문서" 내에서 중요한거 추출하면 a, the 와 같이 필요 없는 단어를 중요하다고 착각할 수 있음.
5. 데이터 분석

#### 토큰화

##### 원핫 인코딩
![image](https://github.com/rimgosu/DeepLearning/assets/120752098/e449eac1-40d8-49cd-aefa-e33780b6ca85)




##### TF-IDF
개별 문서에서 자주 등장하는 단어에는 높은 가중치를 주되, 모든 문서에 자주 등장하는 단어에는 패널티 <br>
TF : 단어가 각 문서에서 발생한 빈도 <br>
DF : 단어 X가 포함된 문서의 수

##### BOW : BAG OF WORDS
CounterVectorize(단순 카운팅) : 빈도 수 기반
![image](https://github.com/rimgosu/DeepLearning/assets/120752098/214c19d0-7a5a-40b6-9d61-84d7af8065eb)



### 9월 14일 (텍스트마이닝, 딥러닝 개요)
> ex01. TextMining(영화리뷰 감성분석)

#### 전처리 고급기술
```
txt_train = [i.replace(b'<br />', b'') for i in txt_train]
```
- 대입변수명 =[누적하고싶은 결과값 for i in 대상범위]
- b'' : byte 문자열을 의미한다. 인코딩을 하지 않아 속도가 빠른 대신, b''로 써줘야함.


#### CountVectorizer
- 빈도수 기반 벡터화 도구
- 오직 띄어쓰기만을 기준으로 하여 단어를 자른 후에 BOW를 만듦!
```
from sklearn.feature_extraction.text import CountVectorizer

# 단어사전 구축
vect = CountVectorizer()
vect.fit(test_words)
print(vect.vocabulary_)

# 단어사전을 바탕으로 벡터값 바꾸기
vect.transform(test_words)
vect.transform(test_words).toarray() # 확인
```

#### LogisticRegression 
- CountVectorizer에서의 LogisticRegression
```
# 로지스틱 모델 생성
logi = LogisticRegression()
logi.fit(X_train, y_train)
logi.score(X_test,y_test)

# 예측
data = rv_vect.transform(data)
pre = logi.predict(data1)
```



#### TF-IDF
![image](https://github.com/rimgosu/DeepLearning/assets/120752098/c4a64ec7-bfde-47bf-9223-382c1b25c62c)
- TF-IDF(Term Frequency - Inverse Document Frequency)
- 단어의 중요도를 확인할 때 단순 카운트 기반이 아닌, 모든 문서를 확인 한 후에 특정 문서에만 자주 등장하는 단어들은 가중치를 부여하는 방식이다
- TF : 하나의 문서에 등장하는 횟수
- DF : 전체의 문서에 등장하는 횟수
- 결과값이 클수록 중요도가 높은단어, 결과값이 낮을수록 중요도가 낮은 단어!

```
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer()
tfidf_vect.fit(corpus) # corpus : 문자열 데이터

tfidf_vect.vocabulary_ # 단어사전 보기
tfidf_vect.transform(corpus).toarray() # 벡터화수치 보기

```


#### 딥러닝 개요
![image](https://github.com/rimgosu/DeepLearning/assets/120752098/e9e2de28-213c-476a-9f61-7cb60b9a9468)
![image](https://github.com/rimgosu/DeepLearning/assets/120752098/277c10f3-bfa3-487a-a0e8-23f6202f3034)




### 9월 15일 
> ex00. 딥러닝 맛보기.ipynb

#### 교차엔트로피 (cross entrophy)

![image](https://github.com/rimgosu/DeepLearning/assets/120752098/0fa36354-4cd7-41b2-90bf-a87cc95540eb)


- 회귀에서는 오차를 MSE로 분류하지만, 분류에서 오차를 교차엔트로피를 활용해 구한다.
- └ (틀릴 때 오차를 증폭시키는 함수를 이용한다.)



#### 딥러닝 프레임워크

1. tensorflow
   - 구글이 만든 딥러닝을 위한 라이브러리

2. keras
   - tensorflow 위에서 동작하는 라이브러리 (사용자 친화적 = 쉬움)
  

> 케라스(Keras)는 TensorFlow, Theano, CNTK 등 딥 러닝 라이브러리를 백엔드로 사용하여 쉽게 다층 퍼셉트론 신경망 모델, 컨볼루션 신경망 모델, 순환 신경망 모델, 조합 모델 등을 구성할 수 있다. <br><br>
>2017년, 구글은 TensorFlow 2.0부터는 코어 레벨에서 Keras를 지원하도록 변경하겠다고 발표하였고, 현재 발표된 TensorFlow 2.0 stable부터는 사실상 전부 Keras를 통해서만 동작하도록 바뀌었다. 사용자용 튜토리얼 페이지 1.15부터 deprecated 목록에 들어가 있던 자잘한 API가 대부분 정리되었고, 익숙되면 조금 더 편하게 사용할 수 있게 변했다. 하지만 그동안 익숙하게 사용해 왔던 모델을 만든 다음 session을 만들어 동작하는 구조에 익숙하던 사람들에게 멘붕을 준 것은 덤.


#### colab
- 마운트 필요.
- 마운트 해야 파일 드라이브에 저장해두고 영구적으로 사용 가능.
- mount 성공 시 "drive" 폴더가 생성됨.

1. 마운트 버튼 클릭
![image](https://github.com/rimgosu/DeepLearning/assets/120752098/e301ec7c-48bc-4097-b206-6f4a068d569d)

2. 다음 코드 실행
```
from google.colab import drive
drive.mount('/content/drive')
```


#### colab 단축키
- 실행단축키
  - ctrl + Enter : 실행 후 커서가 그대로 위치
  - shift + Enter : 실행 후 커서 아래 셀로 이동
  - alt + Enter : 실행 후 아래 셀 생성 아래셀 이동

- 마크다운 변환 (코드 -> 텍스트)
  - ctrl + m + m

- 코드 모드로 변환 (텍스트 -> 코드)
  - ctrl + m + y

- 셀 아래에 추가하기
  - ctrl + m + b

- 셀 위에 추가하기
  - ctrl + m + a

* 더 많은 단축키 : 도구 - 단축키



#### tensorflow
```
# tensorflow 버전 확인
import tensorflow as tf
print(tf.__version__)
```

```
# %cd : 작업 경로를 영구히 바꿈.
%cd "/content/drive/MyDrive/Colab Notebooks/DeepLearning(Spring)"
```

```
# delimiter=';' : 구분자 표시
data = pd.read_csv('./data/student-mat.csv', delimiter=';')
```


#### 기존 머신러닝 (성적데이터)
```
# 기존 머신러닝 모델 흐름
# 1. 모델 생성
linear_model = LinearRegression()
# 2. 모델 학습
linear_model.fit(X_train.values.reshape(-1,1), y_train)
# 3. 모델 예측
linear_pre = linear_model.predict(X_test.values.reshape(-1,1))
# 4. 모델 평가
mean_squared_error(y_test, linear_pre)
```




### 9월 18일
> ex00. 딥러닝 맛보기.ipynb


#### 딥러닝 (성적데이터)
```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Activation
```
1. 신경망 구조 설계
```
# 뼈대 생성
model = Sequential()
# 입력층
model.add(InputLayer(input_shape=(1,))) 
# 중간층 (은닉층)
model.add(Dense(units=10))
model.add(Activation('sigmoid'))
# 출력층
model.add(Dense(units=1))
```



2. 학습 및 평가 방법 설계
```
from scipy import optimize
model.compile(
    loss='mean_squared_error', 
    optimizer='SGD', 
    metrics=['mse']
)
```



3. 모델학습
```
model.fit(X_train, y_train, validation_split=0.2, epochs=20) 
```


4. 모델평가
```
model.evaluate(X_test,y_test)
```





### 9월 18일

> ex01. 유방암 데이터 분류(이진분류).ipynb


```
model = Sequential()

# 입력층 - 학습 데이터의 컬럼이 30개라면 정확하게 "30"으로 지정해주어야 함model.add(InputLayer(input_shape=(30,)))

# 중간층 (은닉층), units과 activation을 동시에 줄 수 있다.
model.add(Dense(units=16, activation='sigmoid'))
model.add(Dense(units=8, activation='sigmoid'))

# 출력층 : 출력받고 싶은 데이터의 형태를 지정함(이진분류 1개의 확률값 0~1)
model.add(Dense(units=1, activation='sigmoid'))
```

#### 퍼셉트론
> 선형모델 + 활성화함수

* 활성화 함수 (중간층, 출력층에서 사용)
* 중간층 : 활성화/ 비활성화(역치)
* 스텝 펑션 => 시그모이드 (왜? 최적화알고리즘 경사하강법을 적용하기 위해 기울기와 역치개념을 가지는 시그모이드 사용)

- 출력층 : 최종 결과의 형태를 결정 <br> (내가 출력하고자 하는 형태에 따라 다르게 작성, units/activation)

```
# 손실함수로 cross entropy 사용
# 이진분류이기 때문에 binary_crossentropy
from scipy import optimize
model.compile(
    loss='binary_crossentropy',
    optimizer='SGD',
    metrics=['accuracy']
)
```

