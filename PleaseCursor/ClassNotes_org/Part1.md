
# 📘 **1. Machine Learning 기본 개념 (Week 2 — Introducing ML)**

## **① 머신러닝의 목적**

* **데이터 기반 일반화(generalization)**
* **패턴 탐지 및 예측 자동화**

## **② 머신러닝 종류**

* **지도학습**: 분류(Classification), 회귀(Regression)
* **비지도학습**: 군집(Clustering), 차원축소(Dimensionality reduction)
* **강화학습**: 보상 기반 학습

## **③ 머신러닝 모델 개념**

* **모델(Model)** = 현실을 단순화한 수학적 구조
* 입력 → 규칙 학습 → 예측

## **④ 머신러닝 프로젝트 절차(Workflow)**

1. 문제 정의
2. 데이터 수집
3. 데이터 정제 및 전처리
4. 알고리즘 선택
5. 모델 학습
6. 모델 평가 및 튜닝
7. 적용(Deployment)

## **⑤ 알고리즘 선택 기준**

* 데이터 형태
* 문제 유형(분류/회귀/군집)
* 해석력 vs 성능
* 데이터 양·품질

---

# 📘 **2. 데이터 관리 및 이해 (Week 3 & Week 4 — Managing & Understanding Data)**

## **① R 데이터 구조**

* **Vector** (동일 타입 1차원)
* **Factor** (범주형 표현: levels)
* **List** (혼합 타입 가능)
* **Data Frame** (표 형식의 데이터)
* **Matrix / Array** (수치형 행렬)

## **② 데이터 전처리 기법**

* **결측치 처리**: 제거, 대체
* **타입 변환**: 숫자/문자/범주형
* **서브셋팅**: 행/열 추출
* **정렬**
* **탐색적 요약(EDA)**: summary(), structure() 등

## **③ 데이터 품질 이슈**

* 이상치(outlier)
* 스케일 차이
* 불균형 데이터
* 변수 간 상관성

---

# 📘 **3. 탐색적 데이터 분석 (Week 10 — EDA)**

## **① EDA 목적**

* 패턴 발견
* 이상치·분포 확인
* 변수 간 관계 탐색

## **② 핵심 분석 기법**

* **요약 통계**
* **히스토그램 / 박스플롯**
* **산점도(Scatterplot)**
* **상관 분석(Correlation analysis)**
* **그룹 비교(집단별 평균·분포 비교)**

---

# 📘 **4. 분류(Classification) (Week 7 — Classification Tree)**

## **① 분류 문제 개념**

* 입력 X로부터 범주형 Y를 예측하는 문제

## **② 의사결정나무(Decision Tree) 핵심 개념**

* **분할(split)**
* **불순도(impurity)**

  * Gini index
  * Entropy
* **정보획득(Information Gain)**
* **가지치기(pruning)**: 과적합 방지
* **트리 깊이(depth)**: 복잡도 조절

---

# 📘 **5. 모델 평가 & 과적합 (Week 7 — Evaluation & Overfitting)**

## **① 데이터 분할 기법**

* **Training / Validation / Test split**
* **Cross-Validation (k-fold)**

## **② 분류 모델 성능 지표**

* **Accuracy**
* **Precision / Recall / F1-score**
* **Confusion Matrix**

## **③ ROC & AUC**

* 이진 분류 성능 평가
* 민감도(Sensitivity) vs FPR 관계 시각화

## **④ 과적합(Overfitting) 개념**

* 학습 데이터에는 잘 맞으나 일반화 성능이 떨어지는 현상

## **⑤ 과적합 방지 기법**

* 모델 규제(regularization)
* 가지치기(pruning)
* 데이터 확충
* 교차검증 활용
* 단순 모델 선택

---

# 📘 **6. k-NN (Week 9 — k Nearest Neighbors)**

## **① k-NN 개념**

* 가장 가까운 k개의 이웃을 기반으로 분류/회귀
* **거리 기반 알고리즘**

## **② 주요 기법**

* 거리측정: **Euclidean distance**
* k 값 선택
* 스케일 표준화 필요
* 다수결(분류), 평균(회귀)

## **③ 장단점**

* 단순·직관적
* 계산량 높음
* 고차원에서 성능 저하(차원의 저주)

---

# 📘 **7. 데이터마이닝 소개 (Week 7 — Data Mining Intro)**

## **① 데이터마이닝 목적**

* 대량 데이터로부터 유용한 패턴/지식 발견

## **② 데이터마이닝 주요 기법(슬라이드 기반)**

* 분류(Classification)
* 회귀(Regression)
* 군집(Clustering)
* 연관규칙(Association Rules)
* 이상치 탐지(Anomaly detection)

## **③ 데이터마이닝 프로세스**

* 문제 정의
* 데이터 준비
* 모델링
* 평가
* 배포
  (KDD 프로세스 개념 포함)

---

