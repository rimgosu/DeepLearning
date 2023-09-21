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
> [ex00. 딥러닝 맛보기.ipynb](https://github.com/rimgosu/ColabBackup/blob/master/ex00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EB%A7%9B%EB%B3%B4%EA%B8%B0.ipynb)

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
> [ex00. 딥러닝 맛보기.ipynb](https://github.com/rimgosu/ColabBackup/blob/master/ex00_%EB%94%A5%EB%9F%AC%EB%8B%9D_%EB%A7%9B%EB%B3%B4%EA%B8%B0.ipynb)


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

> [ex01. 유방암 데이터 분류(이진분류).ipynb](https://github.com/rimgosu/ColabBackup/blob/master/ex01_%EC%9C%A0%EB%B0%A9%EC%95%94_%EB%8D%B0%EC%9D%B4%ED%84%B0_%EB%B6%84%EB%A5%98(%EC%9D%B4%EC%A7%84%EB%B6%84%EB%A5%98).ipynb)


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

### 9월 19일
> [ex02. 손글씨데이터 분류.ipynb](https://github.com/rimgosu/ColabBackup/blob/master/ex02_%EC%86%90%EA%B8%80%EC%94%A8%EB%8D%B0%EC%9D%B4%ED%84%B0_%EB%B6%84%EB%A5%98.ipynb)

#### 출력 형태에 따른 unit의 갯수
- 회귀 : units =1
- 이진분류 : units = 1
- 다중분류 : units = 클래스의 개수

#### 출력 형태에 따른 활성화함수의 종류
- 회귀 : activation='linear' (항등함수, y=x 선형 모델이 예측한 데이터를 그대로 출력) 기본값, 적어주지 않아도 괜찮다
- 이진분류 : activation='sigmoid' (0~1 사이의 확률값을 출력)
- 다중분류 : activation='softmax' (클래스의 개수만큼 확률값을 출력 => 각각의 확률값의 총합이 1이 되도록 출력)
<table><thead><tr><th>출력 형태</th><th>활성화 함수</th><th>손실 함수</th><th>설명</th></tr></thead><tbody><tr><td>회귀</td><td>Linear</td><td>Mean Squared Error (MSE)</td><td>항등함수를 사용한 회귀 모델에 주로 사용</td></tr><tr><td>이진 분류</td><td>Sigmoid</td><td>Binary Cross-Entropy (BCE)</td><td>0과 1 사이의 확률값을 출력하는 이진 분류 모델에 사용</td></tr><tr><td>다중 분류</td><td>Softmax</td><td>Categorical Cross-Entropy</td><td>다중 클래스 분류 모델에서 클래스 간의 교차 엔트로피 사용</td></tr></tbody></table>


### 9월 20일
> [ex02. 손글씨데이터 분류.ipynb](https://github.com/rimgosu/ColabBackup/blob/master/ex02_%EC%86%90%EA%B8%80%EC%94%A8%EB%8D%B0%EC%9D%B4%ED%84%B0_%EB%B6%84%EB%A5%98.ipynb)

#### 범주형 데이터(Y) 학습
```
h1 = digit_model.fit(X_train,y_train,
                     validation_split=0.2,
                     epochs = 20)
```
> ValueError: Shapes (32, 1) and (32, 10) are incompatible


- 방법1: 정답데이터를 확률로 변경

정답 데이터를 확률 값으로 변경하는 방법입니다. 정답 데이터는 현재 클래스 중 하나를 나타내는 형태로 되어 있을 것이며, 모델의 출력은 10개의 클래스에 대한 확률값을 출력하고 있습니다. 이 두 형태를 호환시키기 위해 정답 데이터를 확률로 변환하여 모델과 비교할 수 있습니다. 이를 위해 원-핫 인코딩을 사용하거나 정답 클래스에 해당하는 확률을 1로, 나머지 클래스에는 0으로 설정하는 방법을 고려할 수 있습니다.
```
from tensorflow.keras.utils import to_categorical

one_hot_y_train = to_categorical(y_train)
one_hot_y_train[0:2]
```


- 방법2: loss 함수를 변경

Keras는 다양한 loss 함수를 제공하며, 모델을 학습할 때 사용하는 loss 함수를 변경하여 이 문제를 해결할 수 있습니다. 예를 들어, categorical cross-entropy loss 함수를 사용하면 모델이 10개의 클래스에 대한 확률 분포를 출력하도록 모델을 학습할 수 있습니다. 이러한 loss 함수를 사용하면 모델이 확률로 출력하도록 자동으로 조정됩니다.
따라서, 위의 오류를 해결하기 위해 두 가지 방법 중 하나를 선택하여 모델을 수정하면 됩니다.
```
digit_model.compile(loss = 'sparse_categorical_crossentropy',
                    optimizer= 'SGD',
                    metrics = ['accuracy'])
```


#### PIL
- 파이썬 이미지 처리 라이브러리
```
# 파이썬에서 이미지를 처리하는 라이브러리
import PIL.Image as pimg
```

```
# 이미지 오픈, 흑백이미지로 변경
img = pimg.open('/content/drive/MyDrive/Colab Notebooks/DeepLearning(Spring)/손글씨/0.png').convert('L')
plt.imshow(img, cmap = 'gray')
```

```
# predict
digit_model.predict(test_img)
```



#### 오차역전파

<table><thead><tr><th style="text-align: center;"><strong>과정</strong></th><th style="text-align: center;"><strong>설명</strong></th><th style="text-align: center;"><strong>목적</strong></th></tr></thead><tbody><tr><td style="text-align: center;">순전파 (Forward Propagation)</td><td style="text-align: center;">입력 데이터를 입력층에서 출력층까지 전달하여 예측 값을 계산하는 과정</td><td style="text-align: center;">출력 값을 <strong>추론</strong>하기 위함</td></tr><tr><td style="text-align: center;">역전파 (Backpropagation)</td><td style="text-align: center;">출력층에서 발생한 에러를 입력층까지 전파하여 가중치를 조정하는 과정</td><td style="text-align: center;">모델을 <strong>학습</strong>하여 최적의 결과 도출</td></tr></tbody></table>

##### sigmoid 함수의 문제점
- 기울기 소실 문제 발생
- 한 번 층을 옮길 때마다 1/4(0.25)로 기울기가 줄어든다.

![image](https://github.com/rimgosu/DeepLearning/assets/120752098/ee8b9e16-d6ab-4403-aad4-37b9215a98f0)


![image](https://github.com/rimgosu/DeepLearning/assets/120752098/d55762aa-c8b4-4828-b1b9-4ba11c5e1d39)


##### Relu
- 기울기 소실 문제 해결


