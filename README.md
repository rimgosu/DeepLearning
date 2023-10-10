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


## 딥러닝 개요
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

<table><thead><tr><th>기능</th><th>단축키</th></tr></thead><tbody><tr><td>실행 (커서 그대로 위치)</td><td>Ctrl + Enter</td></tr><tr><td>실행 후 아래 셀로 이동</td><td>Shift + Enter</td></tr><tr><td>실행 후 아래 셀 생성 및 이동</td><td>Alt + Enter</td></tr><tr><td>마크다운으로 셀 변환 (코드 -&gt; 텍스트)</td><td>Ctrl + M, M</td></tr><tr><td>코드 모드로 셀 변환 (텍스트 -&gt; 코드)</td><td>Ctrl + M, Y</td></tr><tr><td>아래에 셀 추가</td><td>Ctrl + M, B</td></tr><tr><td>위에 셀 추가</td><td>Ctrl + M, A</td></tr><tr><td>모든 단축키 목록 보기</td><td>Ctrl + M, H</td></tr></tbody></table>



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

![img](https://github.com/rimgosu/DeepLearning/assets/120752098/1945462d-5fa8-44f5-8843-cd08ebabb52b)

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



### 오차역전파

<table><thead><tr><th style="text-align: center;"><strong>과정</strong></th><th style="text-align: center;"><strong>설명</strong></th><th style="text-align: center;"><strong>목적</strong></th></tr></thead><tbody><tr><td style="text-align: center;">순전파 (Forward Propagation)</td><td style="text-align: center;">입력 데이터를 입력층에서 출력층까지 전달하여 예측 값을 계산하는 과정</td><td style="text-align: center;">출력 값을 <strong>추론</strong>하기 위함</td></tr><tr><td style="text-align: center;">역전파 (Backpropagation)</td><td style="text-align: center;">출력층에서 발생한 에러를 입력층까지 전파하여 가중치를 조정하는 과정</td><td style="text-align: center;">모델을 <strong>학습</strong>하여 최적의 결과 도출</td></tr></tbody></table>


#### sigmoid, relu
<table><thead><tr><th>함수</th><th>특징</th></tr></thead><tbody><tr><td>Step Function</td><td>- 미분 불가능 (기울기가 없음)</td></tr><tr><td></td><td>- 경사 하강법에 사용 불가능</td></tr><tr><td></td><td>- 입력값에 따라 0 또는 1을 출력 (이진 분류에 적합)</td></tr><tr><td>Sigmoid Function</td><td>- 매끄러운 곡선 형태</td></tr><tr><td></td><td>- 미분 가능하지만 기울기 소실 문제 발생</td></tr><tr><td></td><td>- 입력값이 크거나 작을 때 기울기가 0에 가깝게 수렴하여 학습이 어려워질 수 있음</td></tr><tr><td></td><td>- 확률값을 출력하는 데 주로 사용됨</td></tr><tr><td>ReLU (Rectified Linear Unit)</td><td>- 미분 가능하며 기울기 소실 문제를 줄일 수 있음</td></tr><tr><td></td><td>- 입력값이 양수인 경우 그대로 출력하고 음수인 경우 0으로 출력</td></tr><tr><td></td><td>- 학습이 빠르고 효과적일 수 있으며, 주로 은닉층의 활성화 함수로 사용됨</td></tr></tbody></table>

##### Sigmoid
- 'step function'은 경사하강법을 사용하지 못함(기울기 없기 때문)
- 그러나 미분할 때 기울기 소실 문제 발생
- 한 번 층을 옮길 때마다 1/4(0.25)로 기울기가 줄어든다.

![image](https://github.com/rimgosu/DeepLearning/assets/120752098/ee8b9e16-d6ab-4403-aad4-37b9215a98f0)


![image](https://github.com/rimgosu/DeepLearning/assets/120752098/d55762aa-c8b4-4828-b1b9-4ba11c5e1d39)


##### Relu
- 기울기 소실 문제 해결

```
// 손글씨 'sigmoid' => 'relu'
digit_model = Sequential()
digit_model.add(InputLayer(input_shape = (28,28)))
digit_model.add(Flatten()) 
digit_model.add(Dense(units=16, activation='relu'))
digit_model.add(Dense(units=8, activation='relu'))
digit_model.add(Dense(units=32, activation='relu'))
digit_model.add(Dense(units=16, activation='relu'))
digit_model.add(Dense(units=8, activation='relu'))
# 출력층
digit_model.add(Dense(units=10, activation='softmax'))
```







### 9월 22일
> [ex03. 활성화함수, 최적화함수, callback함수.ipynb](https://colab.research.google.com/drive/1wQuM0n3Q1EyHuXU2UljGTsE2HMkbxKgN)
#### 최적화 함수(optimizer)

![img (1)](https://github.com/rimgosu/DeepLearning/assets/120752098/0bda6ea7-bc2b-4846-bfe1-2f36a564116f)


1. 경사하강법
   - 전체 데이터를 이용해 업데이트


2. 확률적경사하강법 (Stochastic Gradient Descent
   - 확률적으로 선택된 일부 데이터를 이용해 업데이트
   - 경사하강법보다 더 빨리, 더 자주 업데이트한다.

3. 모멘텀
   - 관성을 적용해 업데이트 현재 batch 뿐만 아니라 이전 batch 데이터의 학습 결과도 반영
   - `α : learning rate, m : 모멘텀 계수`

4. 네스테로프 모멘텀 (NAG)
   - 미리 해당 방향으로 이동한다고 가정하고 기울기를 계산해본 뒤 실제 업데이트 반영

5. adam
   - 요즘 가장 각광받는 최적화 함수


<table><thead><tr><th>조합</th><th>특징</th></tr></thead><tbody><tr><td><strong>sigmoid + SGD</strong></td><td>- 아주 낮은 정확도를 보임. &lt;br&gt; - Sigmoid 함수는 그래디언트 소실 문제가 발생할 수 있어, 딥 뉴럴 네트워크에서는 잘 활용되지 않습니다. &lt;br&gt; - SGD는 학습 속도가 느릴 수 있습니다.</td></tr><tr><td><strong>relu + SGD</strong></td><td>- 중간 정도의 정확도를 보임. &lt;br&gt; - ReLU는 그래디언트 소실 문제에 덜 민감하지만, SGD를 사용하면 여전히 학습 속도가 느릴 수 있습니다.</td></tr><tr><td><strong>relu + Adam</strong></td><td>- 가장 높은 정확도를 보임. &lt;br&gt; - ReLU와 Adam 조합은 그래디언트 소실 문제를 효과적으로 방지하고, Adam 최적화 알고리즘은 학습률 조절로 빠른 학습을 지원합니다.</td></tr></tbody></table>

![image](https://github.com/rimgosu/DeepLearning/assets/120752098/6735ea4c-0ad3-4b56-8a11-6c396d2e5f92)




##### Batch_size
- 일반적으로 PC 메모리의 한계 및 속도 저하 때문에 대부분의 경우에는 한번의 epoch에 모든 데이터를 한꺼번에 집어넣기가 힘듦
- `epochs 당 학습 : train_data_size/batch_size `
- batch size 32 : 1500/1500 => batch size 128 : 375/375

![image](https://github.com/rimgosu/DeepLearning/assets/120752098/39e6a0db-c051-4950-be15-e8936f1635f6)


##### callback 함수
- 모델저장 및 조기학습중단
- 모델저장
  - 딥러닝모델 학습시 지정된 epoch 를 다 끝내면 과대적합이 일어나는 경우가 있다 -> 중간에 일반화된 모델을 저장할 수 있는 기능!!
- 조기학습 중단
  - epoch 를 크게 설정할 경우 일정횟수 이상으로는 모델의 성능이 개선되지 않는 경우가 있다.  -> 시간낭비 -> 모델의 성능이 개선되지 않는 경우에는 조기중단이 필요

1. 콜백함수 임포트 <br>
`from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping`

2. hdf5 파일(모델 로그) 경로 작성
```
model_path = '/content/drive/MyDrive/Colab Notebooks/DeepLearning(Spring)/data/digit_model/dm_{epoch:02d}_{val_accuracy:0.2f}.hdf5'
```

3. ModelCheckpoint

```
ModelCheckpoint(
    filepath = model_path,
    verbose = 1, 
    save_best_only = True,
    monitor = 'val_accuracy'
)
```

4. EarlyStopping

```
model_path = '/content/drive/MyDrive/Colab Notebooks/DeepLearning(Spring)/data/digit_model/dm_{epoch:02d}_{val_accuracy:0.2f}.hdf5'
mckp = ModelCheckpoint(
    filepath = model_path,
    verbose = 1, # 로그 출력, 1: 로그 출력 해주세요
    save_best_only = True, # 모델 성능이 최고점을 갱신할 때마다 저장
    monitor = 'val_accuracy' # 최고점의 기준치
)
```

5. 학습 (callbacks = [mckp, early] 파라미터 추가)

```
h4 = model3.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs = 1000,
    batch_size=128,
    callbacks = [mckp, early]
)
```

6. 파일로 저장된 모델 불러오기
```
from tensorflow.keras.models import load_model

best_model = load_model('/content/drive/MyDrive/Colab Notebooks/DeepLearning(Spring)/data/digit_model/dm_19_0.73.hdf5')
``` 




### 9월 25일

![image](https://github.com/rimgosu/DeepLearning/assets/120752098/39c4440e-8218-4c23-b291-86f133b74e6b)


> [dogs_vs_cats_small](https://drive.google.com/drive/folders/1ciEeGooDZsZZ81UJibvSAUaLB1qi42ih) <br>
> [ex04. 개 고양이 분류하기.ipynb](https://colab.research.google.com/drive/1AK7WZ7W1q4oUMXKpYf_LLJQBZHjGPmJo) <br>

#### CNN(Convolution Neural Network)

![image](https://github.com/rimgosu/DeepLearning/assets/120752098/82c841b6-afd7-42d9-a900-6bd9ad259c5d)

1. 이미지 학습 가능 (2차원의 데이터도 학습)
2. 데이터의 특징을 추출하고 추출된 특징을 기반으로 학습
   - 이미지의 크기, 방향 등에 크게 관여하지 않는다
3. CNN 층은 CONV LAYER, POOLING LAYER 있다.

| Layer   | Function                                         |
|---------|--------------------------------------------------|
| CONV    | 특징을 찾는다.                                   |
| POOLING | 특징이 아닌 부분을 찾아 해상도를 낮춘다.          |
| DENSE   | 찾아진 특징을 토대로 사물을 구분할 규칙을 만든다. |


#### cat vs dog 실습

a. ImageDataGenerator
   - 이미지 데이터를 생성하고 증강하기 위한 도구를 제공
   - 이미지 데이터를 부풀리기 위해 사용

```
# 하나의 변수에 이미지 파일 전부다 합치기
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

   1. 픽셀값 정규화 (0~255 → 0~1)
```
generator = ImageDataGenerator(rescale= 1./255)
```

   2. 하나의 변수에 이미지 파일 전부 담기
      - target_size = (150,150) : 변환할 이미지의 크기
      - class_mode = 'binary' : 라벨링 방법, 다중분류 : categorical
```
train_generator = generator.flow_from_directory(
    directory = train_dir,
    target_size = (150,150),
    batch_size = 100,
    class_mode = 'binary'
)

valid_generator = generator.flow_from_directory(
    directory = valid_dir,
    target_size = (150,150),
    batch_size = 100, 
    class_mode = 'binary'
)
```

b. 모델 설계를 위한 라이브러리 불러오기
```
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
```

c. CNN 모델 설계
   1. filter = 32 : 찾을 특징의 갯수
   2. kernel_size = (3,3) : 특징의 크기
   3. input_shape = (150,150,3) : 입력 데이터의 모양 (가로, 세로, RGB) 
      - 0 : 검은색, 255 : 흰색
      - 입구 : 첫 번째 Conv2D에만 지정해준다.

```
model1 = Sequential()

model1.add(Conv2D(
    filters = 32,
    kernel_size = (3,3),
    input_shape = (150,150,3),
    activation = 'relu'
))

model1.add(MaxPool2D(
    pool_size= (2,2)
))

model1.add(Conv2D(
    filters = 32,
    kernel_size = (3,3),
    activation = 'relu'
))

model1.add(MaxPool2D(
    pool_size= (2,2)
))
```

d. 분류분석 & 출력
   1. model1.add(Flatten()) : 특징추출부와 분류부를 이어주는 역할
   2. model1.add(Dense(units = 32, activation = 'relu')) : Dense층 추가
   3. model1.add(Dense(units= 1, activation = 'sigmoid')) : 출력층
      - 이진 분류를 해야되므로 sigmoid를 쓴다.
```
model1.add(Flatten())
model1.add(Dense(units = 32, activation = 'relu'))
model1.add(Dense(units= 1, activation = 'sigmoid'))
```

e. 학습 방법 설정 (compile)
```
model1.compile(
    loss = 'binary_crossentropy',
    optimizer = 'Adam',
    metrics = ['accuracy']
)
```

f. 학습하기 (fit)
   1. train_generator : 학습데이터 (X_train, y_train 합쳐져 있다)
   2. valid_generator : 검증데이터 (X_val, y_val) 
```
model1.fit(
    train_generator, 
    epochs = 20,
    validation_data = valid_generator
)
```


### 9월 27일

![image](https://github.com/rimgosu/DeepLearning/assets/120752098/fd85d080-ac41-48b1-b642-7b6548e39f97)

> [ex04. 개 고양이 분류하기.ipynb](https://colab.research.google.com/drive/1AK7WZ7W1q4oUMXKpYf_LLJQBZHjGPmJo) <br>

a. [image-kernels](https://setosa.io/ev/image-kernels/)

- filter를 보면 _축소샘플링_ 으로인해 가장자리가 죽은 것을 볼 수 있다.

![image](https://github.com/rimgosu/DeepLearning/assets/120752098/e351a7b9-b6cd-43e8-aff2-e974a748bcc8)

b. 축소샘플링
- 축소 샘플링을 진행할 때 : `padding='valid'`
- 축소 샘플링을 진행 안할 때 : `padding='same'`
   - 이 때 원본 이미지의 가장자리를 0으로 채워넣어 작아질 만큼 이미지 크기를 키운다
 


### 10월 4일

> [ex04. 개 고양이 분류하기.ipynb](https://colab.research.google.com/drive/1AK7WZ7W1q4oUMXKpYf_LLJQBZHjGPmJo#scrollTo=MN2pq86e8D2P)

#### 과대적합을 줄이는  방법들

- 증식
  - 장점 : 간단
  - 단점 : 가짜는 가짜다. (급격한 성능 향상은 없다)
  - epoch 수는 증가시켜주어야한다.
- dropout()
  - 층에 사용하는 퍼셉트론의 수를 설정한 비율만큼 사용하지 않는 방법
  - epoch마다 사용하지 않는 퍼셉트론은 랜덤 
- BatchNormalization() : 정규화
  - CNN층 = Conv (특성추출) + Maxpooling (크기 축소)
  - Conv층의 파라미터를 정규화(평균 0, 분산 1) <br>
  => 음수가 발생 <br>
  => relu를 적용하면 <br>
  => 음수가 사라지는 문제 => leaky relu
- GlobalAveragePooling2D()
  - CNN에서 가장 문제가 되는 층 : Maxpooling 층 <br>
  => CNN 속도의 60% 이상을 차지 (느리다)
  - Maxpooling2D+ Flatten()
  - Dense 층과 연결되는 층에 사용

#### Dropout
```
from tensorflow.keras.layers import Dropout, BatchNormalization, GlobalAveragePooling2D
```

- Dropout 층은 파라미터 수 차이가 커서 과적합이 발생하기 쉬운 층과 층 사이에 집어넣는다.
```
# 모델의 층, 파라미터 수 확인
model1.summary()
```

```
# Dropout 층 추가
model1.add(Dropout(0.5))

# 드롭아웃 층을 넣고 모델을 학습하면 과적합이 살짝 줄어든 것을 볼 수 있다.
```


#### GlobalAveragePooling

- 층을 추가하면 파라미터간의 차이가 확 줄어드는 것을 볼 수있다.
```
# GlobalAveragePooling 층 추가
model1.add(GlobalAveragePooling2D())

# 이 층으로 학습을 진행하면 정확도는 매우 떨어지지만, 과적합이 상당히 개선된 것을 볼 수 있다.
```


#### BatchNormalization
- Conv2D()와 Activation() 층 사이에 배치
- Activation()이 정규화 기능을 일부 수행 => Activation() 다음에 배치하면 효과가 떨어짐
- 원칙 : C + B + A + M => 근데 일반적으로 : C + A + B + M
- 성능이 확실하게 향상되는 게 장점이다.
- 단점 : 왜 좋아지는지 알 길이 없다.




#### 전이학습
![image](https://github.com/rimgosu/DeepLearning/assets/120752098/72bc4af4-9b98-4f28-867b-695c65507009)

- 전이학습 하는 이유
  - 데이터가 부족
  - 설계한 신경망이 그닥 좋지 못하다

- 전이학습의 종류
  - 특성 추출 : CNN층의 가중치를 그대로 사용
  - 미세 조정 (fine tuning) : CNN층의 가중치를 일부 살짝 변경해서 사용

![image](https://github.com/rimgosu/DeepLearning/assets/120752098/2f5dc397-008f-4c61-88e8-12563682bd47)




### 10월 6일
#### 전이학습-2

> [ex04. 개 고양이 분류하기.ipynb](https://colab.research.google.com/drive/1AK7WZ7W1q4oUMXKpYf_LLJQBZHjGPmJo#scrollTo=MN2pq86e8D2P)

##### VGG 다운로드

```
from tensorflow.keras.applications import VGG16

# weights : 사용할 가중치의 종류 (imagenet)
# include_top = False : 모델을 전체 또는 특성추출기만 가져올 것인지 선택 (False : 특성추출기만 가져옴)
conv_base = VGG16(
    weights = "imagenet",
    include_top = False,
    input_shape = (150, 150, 3)
)
```

##### 모델을 시각화하는 두 가지 방법
1. summary
```
모델.summary()
```
2. plot_model
   - summary 방식보다 좀 더 직관적임
```
plot_model(모델, show_shapes= True, dpi=60)
```

3. visual keras
- visual keras 다운로드
  
```
!pip install visualkeras
```

- 구로를 이미지로 시각화
```
import visualkeras

visualkeras.layered_view(모델).show()
visualkeras.layered_view(모델, legend=True)
```

![image](https://github.com/rimgosu/DeepLearning/assets/120752098/360de1bd-31fd-444f-9512-36bd75e40f7e)

_이뿌당.._




##### 모델 이어붙이기
- 기본 모델을 이미 만들어져있는 conv_base 모델 층을 통째로 이어붙여 학습함.
- 구조 : conv_base => 기존에 만들어져있던 모델
```
model1 = Sequential()

model1.add(conv_base)
model1.add(Flatten())

model1.add(Dropout(0.5))
model1.add(Dense(units = 64, activation = LeakyReLU(alpha=0.1)))
model1.add(Dense(units = 128, activation = LeakyReLU(alpha=0.1)))
model1.add(Dense(units = 64, activation = LeakyReLU(alpha=0.1)))
model1.add(Dense(units = 32, activation = LeakyReLU(alpha=0.1)))
model1.add(Dense(units = 1, activation = 'sigmoid'))

model1.summary()
```

- 모델을 이어붙일 때, **동결**시켜주는 과정이 필요하다
- 가져온 모델은 학습이 되지 않도록 설정
- 동결 후 모델 학습하면 성능이 굉장히 향상되는 것을 볼 수 있다.

```
conv_base.trainable = False
```

- fit 결과 : val_accuracy가 89%까지 올라간 것을 볼 수 있다.
```
Epoch 10/10
20/20 [==============================] - 9s 461ms/step - loss: 0.1216 - accuracy: 0.9515 - val_loss: 0.3081 - val_accuracy: 0.8850
```


##### 미세조정(fine tuning)
- 분류기가 연결되는 모델의 층만 학습이 되도록 동결을 풀어주는 것

| 층   | 1번층  | 2번층  | 3번층  | 4번층  | 5번층  |
| ---- | ------ | ------ | ------ | ------ | ------ |
| 학습  | 학습x  | 학습x  | 학습x  | 학습x  | 학습o  |

```
conv_base.trainable = True
set_trainable = False

for layer in conv_base.layers:
  if layer.name == "block5_conv1":
    set_trainable = True

  if set_trainable :
    layer.trainable = True
  else:
    layer.trainable = False
```


##### Xception 모델

![image](https://github.com/rimgosu/DeepLearning/assets/120752098/af695e44-364a-47e2-bfc8-1add9849c020)

- 다음 코드를 작성해서 Xception 모델을 불러올 수 있다
```
from tensorflow.keras.applications import Xception

conv_base2 = Xception(
    weights ="imagenet",
    include_top =False,
    input_shape = (150, 150,3)
)
```

- 시각화 해보면 굉장히 길고 복잡하다는 것을 알 수 있다
- BatchNormalization 층이 추가되어 있는 것을 볼 수 있음.

![image](https://github.com/rimgosu/DeepLearning/assets/120752098/d9f8c692-3e9e-4a11-a725-7ea478777d48)

- 학습 결과 : val_accuracy가 97%로 성능이 매우 올라간 것을 볼 수 있다.
```
Epoch 10/10
20/20 [==============================] - 9s 429ms/step - loss: 9.1577e-04 - accuracy: 1.0000 - val_loss: 0.1350 - val_accuracy: 0.9690
```





## yolov7

![image](https://github.com/rimgosu/DeepLearning/assets/120752098/b352e077-5c34-40dd-b4f3-9a54180ef8a5)


### 10월 10일

> [프로젝트005_YoloV7기반_Custom데이터로_객체탐지.ipynb](https://colab.research.google.com/drive/1idh8nzKG8u0lbPk867Dc8jSbFSSOdkc7)
   - 이 프로젝트에 있는 튜토리얼 똑같이 따라하면 된다.
- Yolov5s, Yolov5m, YOlov5l, Yolov5x 있을 때 현재 컴퓨터론 l,x는 못쓴다.

- [로보플로우 프로젝트](https://app.roboflow.com/lecture-2sthx)
   - 어노테이션 작업을 위해 가장 많이 쓰는 사이트
 



