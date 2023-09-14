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
# 로지스텍 모델 생성
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








