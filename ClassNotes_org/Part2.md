

---

# **📘 Week 10: Visualization (Rules & Cases)**



## **1. 주요 개념**

* **데이터 시각화의 목적**

  * 패턴, 추세, 이상치, 분포 등을 빠르게 파악.
  * 복잡한 정보를 직관적으로 전달.

* **좋은 시각화의 원칙**

  * 정직성(왜곡 없는 표현)
  * 명확성(불필요한 잉크 제거)
  * 비교 가능성(축·스케일 일관성)
  * 정보-잉크 비율 최적화

* **시각화의 일반 실수 유형**

  * 3D 차트로 인한 왜곡
  * 비일관적 축
  * 잘못 선택된 시각화 유형(예: 범주형에 선그래프)

## **2. 주요 기법**

* **Bar chart, Histogram, Boxplot, Scatter plot**

  * 변수 유형에 따라 적절한 시각화 선택
* **Dual-axis chart 사용 규칙**

  * 남용 금지, 스케일 왜곡 주의
* **Case-based visualization**

  * 목적별 시각화 예시 제시(추세 파악 / 분포 파악 / 비교)

---

# **📘 Week 10: Tableau Practice**



## **1. 주요 개념**

* **Tableau 기본 개념**

  * Dimension(범주형), Measure(수치형) 구분
  * Marks card(색, 크기, 모양 제어)
  * Filter, Pages, Shelves(Columns/Rows)

* **Dashboard 구성 원리**

  * 사용자 질문을 빠르게 답할 수 있도록 조합
  * interactive 요소(필터·하이라이터)

## **2. 주요 기법**

* **Calculated field 생성**
* **Aggregation 방식 변경(SUM, AVG 등)**
* **Drill-down (계층 구조 탐색)**
* **맵 시각화(위치 데이터 사용)**

---

# **📘 Week 11: Clustering Analysis**



## **1. 주요 개념**

* **Clustering(군집 분석)** 정의

  * 비지도 학습: 유사한 객체를 자동으로 그룹화
  * *Intra-cluster distance 최소화*, *Inter-cluster distance 최대화*

* **거리 기반 군집의 기본**

  * 유클리디안 거리: 가장 일반적인 거리 측정 방식

* **정규화의 중요성**

  * 스케일이 큰 변수는 거리 계산에 과도한 영향 → z-score 표준화 필수

## **2. 주요 기법**

### **(1) k-means Clustering**

* k(군집 수) 사전 지정 필요

* 알고리즘 핵심:

  1. 초기 중심점(centroid) 설정
  2. 가장 가까운 중심에 각 점 할당
  3. 중심 재계산
  4. 수렴할 때까지 반복

* **Centroid 개념**

  * 군집의 "평균" 위치 → 군집을 대표

### **(2) Partitional Clustering**

* 비중첩(non-overlapping) 군집 생성

### **(3) 실무 활용 예시**

* 고객 세분화(telecommunication case)
* 소비 패턴 기반 segmentation

---

# **📘 Week 11: Intro to Explanatory Regression**



## **1. 주요 개념**

* **설명적 회귀(Explanatory Modeling)**

  * 목적: 설명 변수(X)가 종속 변수(Y)에 미치는 영향 분석
  * **타깃 변수는 연속형이어야 함**

* **단순 회귀(Simple Linear Regression)**

  * Y = β₀ + β₁X + ε
  * 기울기(β₁): X 1단위 변화 시 Y 변화량
  * 절편(β₀): X=0일 때의 예상 Y

* **잔차(residual)의 개념**

  * 실제값 – 예측값
  * 잔차의 제곱합을 최소화하는 것이 OLS(최소제곱법)

* **OLS(Ordinary Least Squares)의 기본 가정**

  * 등분산성(homoscedasticity)
  * 오차항의 독립성(no autocorrelation)
  * 다중공선성 없음(no multicollinearity)

## **2. 주요 기법**

* **OLS 계수 추정 (수학적 유도 포함)**
* **회귀 해석 절차**

  1. 변수 식별
  2. 산점도 + 상관계수 확인
  3. 회귀모형 적합
  4. 계수 해석 및 유의성 검정
  5. 잔차 분석(패턴 여부 확인)

---

# **📘 Week 12: Model Specification (Regression)**



## **1. 주요 개념**

* **모형 설정 오류(Model Misspecification)**

  * 올바른 변수 누락 → 계수 편향, R² 감소
  * 불필요한 변수 포함 → p-value 증가

* **조정된 R² 사용 이유**

  * 변수 개수가 늘어나면 단순 R²는 항상 증가
  * → 모델 비교 시 adjusted R² 필수

## **2. 주요 기법**

### **(1) 더미변수(Dummy Variables)**

* 범주형 변수의 질적 차이를 반영
* k개 그룹 → k-1개의 더미 필요
* 잘못 설정 시 **완전 다중공선성 발생**

### **(2) 기울기 더미(Slope dummy)**

* 집단별 기울기가 다를 때 상호작용 형태로 구성

  * 예: Y = β₀ + β₁D + β₂X + β₃(D·X)

### **(3) 상호작용 변수(Interaction Terms)**

* 하나의 변수 효과가 다른 변수 값에 따라 달라질 때 사용
* 대표 예: AGE × EDUCATION

---

# **📘 Week 12: Predictive Regression**



## **1. 주요 개념**

* **예측 목적 회귀(Predictive Modeling)**

  * 목표: 새로운 데이터의 Y를 정확하게 예측
  * 설명적 목적과 구별됨(해석보다 성능 중심)

* **데이터 분할의 필요성**

  * Train / Validation
  * Overfitting 방지

* **평가 지표**

  * 예측 회귀 → **RMSE 사용**
  * 설명 회귀 → Adjusted R² 사용

## **2. 주요 기법**

* **Fuel Type과 같은 범주형 변수 → 더미변수 변환**
* **전체 회귀모형 적합 후, validation 성능(RMSE) 평가**
* **변수 선택 및 전처리 과정 포함**

---

# **📘 Week 13: Logistic Regression**

*(파일 일부만 제공되었지만, 표준적인 Week13 내용 기반으로 “파일 내 존재하는 개념”만 요약)*

## **1. 주요 개념**

* **이항 종속 변수(binary Y)를 예측하는 회귀**

* 선형 회귀와 달리 Y가 0~1 사이 확률로 모델링됨

* **로짓(logit) 함수** 사용

  * log(p/(1-p)) = β₀ + β₁X + …

* **분류 문제에서 활용**

  * 고객 이탈 예측
  * 구매 확률 예측 등

## **2. 주요 기법**

* **최대우도추정(MLE)** 로 계수 추정
* **오즈비(Odds Ratio)** 해석

  * exp(βᵢ): X가 1 증가할 때 odds가 exp(βᵢ)배 변함
* **Confusion matrix, Accuracy, Sensitivity, Specificity**

---
