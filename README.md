![image](https://github.com/rimgosu/DeepLearning/assets/120752098/e0b6d433-cecb-4397-a2c1-5dbfe2e820d2)


# DeepLearning



## Text Mining

![image](https://github.com/rimgosu/DeepLearning/assets/120752098/1d5fb433-4e4f-4277-b45f-9b402f87611f)

### 9월 13일 (머신러닝 복습, 텍스트 마이닝)
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

원 핫 인코딩 : <br> 1. CounterVectorize(단순 카운팅) <br> 2. TF-IDF(문서내 빈도수 확인)
![image](https://github.com/rimgosu/DeepLearning/assets/120752098/e449eac1-40d8-49cd-aefa-e33780b6ca85)

BOW : BAG OF WORDS
![image](https://github.com/rimgosu/DeepLearning/assets/120752098/214c19d0-7a5a-40b6-9d61-84d7af8065eb)


4. 특징 값 추출(중요한 단어 추출) <br> TF-IDF : "문서"내에서 중요한 단어여야함. <br>"모든문서" 내에서 중요한거 추출하면 a, the 와 같이 필요 없는 단어를 중요하다고 착각할 수 있음.
5. 데이터 분석










