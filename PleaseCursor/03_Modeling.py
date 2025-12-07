# -*- coding: utf-8 -*-
"""
03_Modeling.py
모델링: OLS 회귀분석, K-means 클러스터링, 분류모델
- 점포당 매출액 영향 요인 분석
- 상권 유형 분류 모델
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 통계 모델링
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 머신러닝
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, silhouette_score)
from sklearn.decomposition import PCA

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("모델링: 회귀분석 및 분류")
print("=" * 60)

# ============================================================
# 1. 데이터 로드
# ============================================================
print("\n" + "=" * 60)
print("1. 데이터 로드")
print("=" * 60)

df = pd.read_csv('df_modeling.csv')
print(f"데이터 로드: {df.shape}")

# 결측치 확인
print(f"\n결측치 현황:")
missing = df.isnull().sum()
print(missing[missing > 0])

# 분석용 데이터 준비 (결측치 제거)
df_clean = df.dropna(subset=['점포당_매출액']).copy()
print(f"\n분석 대상: {len(df_clean)} 상권")

# ============================================================
# 2. OLS 회귀분석
# ============================================================
print("\n" + "=" * 60)
print("2. OLS 회귀분석")
print("=" * 60)

# 독립변수 선택
independent_vars = ['총_유동인구_수', '총_상주인구_수', '월_평균_소득_금액',
                    '유사_업종_점포_수', '경쟁강도', '프랜차이즈_비율', '영역_면적']

# 존재하는 변수만 선택
available_vars = [v for v in independent_vars if v in df_clean.columns]
print(f"독립변수: {available_vars}")

# 결측치 처리
df_reg = df_clean[['점포당_매출액', 'log_점포당_매출액'] + available_vars].dropna()
print(f"회귀분석 데이터: {len(df_reg)} 상권")

# 로그 변환된 종속변수 사용 (정규성 개선)
y = df_reg['log_점포당_매출액']
X = df_reg[available_vars]

# 스케일링
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# 상수항 추가
X_const = sm.add_constant(X_scaled)

# OLS 회귀분석
model_ols = sm.OLS(y, X_const).fit()
print("\n" + "=" * 40)
print("OLS 회귀분석 결과")
print("=" * 40)
print(model_ols.summary())

# VIF 계산 (다중공선성 확인)
print("\n다중공선성 검정 (VIF):")
vif_data = pd.DataFrame()
vif_data['변수'] = X_scaled.columns
vif_data['VIF'] = [variance_inflation_factor(X_scaled.values, i) for i in range(X_scaled.shape[1])]
print(vif_data.to_string(index=False))

# VIF 결과 저장
vif_data.to_csv('model_vif_results.csv', index=False, encoding='utf-8-sig')
print("\n[저장] model_vif_results.csv")

# 시각화: 회귀 진단
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 2-1. 잔차 vs 적합값
ax1 = axes[0, 0]
fitted = model_ols.fittedvalues
residuals = model_ols.resid
ax1.scatter(fitted, residuals, alpha=0.5, color='#2E86AB')
ax1.axhline(y=0, color='red', linestyle='--')
ax1.set_xlabel('Fitted Values')
ax1.set_ylabel('Residuals')
ax1.set_title('Residuals vs Fitted', fontsize=12, fontweight='bold')

# 2-2. Q-Q Plot
ax2 = axes[0, 1]
stats.probplot(residuals, dist="norm", plot=ax2)
ax2.set_title('Normal Q-Q Plot', fontsize=12, fontweight='bold')

# 2-3. 잔차 히스토그램
ax3 = axes[1, 0]
ax3.hist(residuals, bins=30, color='#2E86AB', alpha=0.7, edgecolor='white')
ax3.set_xlabel('Residuals')
ax3.set_ylabel('Frequency')
ax3.set_title('Residuals Distribution', fontsize=12, fontweight='bold')

# 2-4. 회귀계수
ax4 = axes[1, 1]
coef = model_ols.params.drop('const')
coef_sorted = coef.sort_values()
colors = ['#2E86AB' if x > 0 else '#E94F37' for x in coef_sorted.values]
ax4.barh(coef_sorted.index, coef_sorted.values, color=colors, alpha=0.8)
ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax4.set_xlabel('Coefficient (Standardized)')
ax4.set_title('Regression Coefficients', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('model_01_ols_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[저장] model_01_ols_diagnostics.png")

# OLS 결과 저장
ols_results = pd.DataFrame({
    '변수': model_ols.params.index,
    '계수': model_ols.params.values,
    '표준오차': model_ols.bse.values,
    't값': model_ols.tvalues.values,
    'p값': model_ols.pvalues.values
})
ols_results.to_csv('model_ols_results.csv', index=False, encoding='utf-8-sig')
print("[저장] model_ols_results.csv")

# ============================================================
# 3. K-means 클러스터링 (횡단면 기반)
# ============================================================
print("\n" + "=" * 60)
print("3. K-means 클러스터링 (횡단면)")
print("=" * 60)

# 클러스터링 변수
cluster_vars = ['점포당_매출액', '총_유동인구_수', '경쟁강도', '프랜차이즈_비율']
cluster_vars = [v for v in cluster_vars if v in df_clean.columns]

df_cluster = df_clean[cluster_vars].dropna()
print(f"클러스터링 데이터: {len(df_cluster)} 상권")

# 스케일링
scaler_cluster = StandardScaler()
X_cluster = scaler_cluster.fit_transform(df_cluster)

# Elbow Method
inertias = []
silhouettes = []
K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_cluster, kmeans.labels_))

# 시각화: Elbow + Silhouette
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
ax1.set_ylabel('Inertia', fontsize=12)
ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(K_range, silhouettes, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Silhouette Score', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_02_kmeans_elbow.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[저장] model_02_kmeans_elbow.png")

# K=5로 클러스터링 (상권유형 5개와 비교)
n_clusters = 5
kmeans_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df_clean.loc[df_cluster.index, 'KMeans_Cluster'] = kmeans_final.fit_predict(X_cluster)

print(f"\nK-means 클러스터 분포 (K={n_clusters}):")
print(df_clean['KMeans_Cluster'].value_counts().sort_index())

# PCA 시각화
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster)

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=kmeans_final.labels_, cmap='viridis', alpha=0.6)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
ax.set_title('K-means 클러스터링 결과 (PCA)', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Cluster')

plt.tight_layout()
plt.savefig('model_03_kmeans_pca.png', dpi=150, bbox_inches='tight')
plt.close()
print("[저장] model_03_kmeans_pca.png")

# 클러스터별 특성
cluster_stats = df_clean.groupby('KMeans_Cluster')[cluster_vars].mean().round(2)
print("\n클러스터별 평균 특성:")
print(cluster_stats)

# 결과 저장
cluster_stats.to_csv('model_kmeans_results.csv', encoding='utf-8-sig')
print("[저장] model_kmeans_results.csv")

# ============================================================
# 4. 분류 모델 (상권유형 예측)
# ============================================================
print("\n" + "=" * 60)
print("4. 분류 모델 (상권유형 예측)")
print("=" * 60)

if '상권유형' in df_clean.columns and df_clean['상권유형'].notna().sum() > 100:
    # 분류용 데이터 준비
    df_class = df_clean[['상권유형'] + available_vars].dropna()
    print(f"분류 데이터: {len(df_class)} 상권")
    
    # 레이블 인코딩
    le = LabelEncoder()
    y_class = le.fit_transform(df_class['상권유형'])
    X_class = df_class[available_vars]
    
    # 스케일링
    X_class_scaled = scaler.fit_transform(X_class)
    
    # 훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_class_scaled, y_class, test_size=0.3, random_state=42, stratify=y_class
    )
    
    print(f"\n훈련 데이터: {len(X_train)}, 테스트 데이터: {len(X_test)}")
    
    # 4-1. 로지스틱 회귀
    print("\n[로지스틱 회귀]")
    lr_model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    print(f"정확도: {accuracy_score(y_test, lr_pred):.3f}")
    print("\n분류 리포트:")
    print(classification_report(y_test, lr_pred, target_names=le.classes_))
    
    # 4-2. 의사결정나무
    print("\n[의사결정나무]")
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42, min_samples_leaf=10)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    
    print(f"정확도: {accuracy_score(y_test, dt_pred):.3f}")
    print("\n분류 리포트:")
    print(classification_report(y_test, dt_pred, target_names=le.classes_))
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 혼동행렬 - 로지스틱 회귀
    ax1 = axes[0]
    cm_lr = confusion_matrix(y_test, lr_pred)
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax1.set_title(f'로지스틱 회귀 혼동행렬\n(정확도: {accuracy_score(y_test, lr_pred):.3f})', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # 혼동행렬 - 의사결정나무
    ax2 = axes[1]
    cm_dt = confusion_matrix(y_test, dt_pred)
    sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens', ax=ax2,
                xticklabels=le.classes_, yticklabels=le.classes_)
    ax2.set_title(f'의사결정나무 혼동행렬\n(정확도: {accuracy_score(y_test, dt_pred):.3f})', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('model_04_classification.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[저장] model_04_classification.png")
    
    # 의사결정나무 시각화
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(dt_model, feature_names=available_vars, class_names=le.classes_,
              filled=True, rounded=True, ax=ax, fontsize=8)
    ax.set_title('의사결정나무 구조', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('model_05_decision_tree.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[저장] model_05_decision_tree.png")
    
    # 변수 중요도
    print("\n변수 중요도 (의사결정나무):")
    importance = pd.DataFrame({
        '변수': available_vars,
        '중요도': dt_model.feature_importances_
    }).sort_values('중요도', ascending=False)
    print(importance.to_string(index=False))
    
    # 분류 결과 저장
    class_results = pd.DataFrame({
        '모델': ['Logistic Regression', 'Decision Tree'],
        '정확도': [accuracy_score(y_test, lr_pred), accuracy_score(y_test, dt_pred)]
    })
    class_results.to_csv('model_classification_results.csv', index=False, encoding='utf-8-sig')
    print("\n[저장] model_classification_results.csv")

else:
    print("상권유형 데이터가 부족하여 분류 모델을 건너뜁니다.")

# ============================================================
# 5. 결과 요약
# ============================================================
print("\n" + "=" * 60)
print("5. 결과 요약")
print("=" * 60)

print("\n[OLS 회귀분석 요약]")
print(f"- R-squared: {model_ols.rsquared:.3f}")
print(f"- Adj. R-squared: {model_ols.rsquared_adj:.3f}")
print(f"- F-statistic: {model_ols.fvalue:.2f} (p={model_ols.f_pvalue:.4f})")

print("\n유의한 변수 (p < 0.05):")
sig_vars = model_ols.pvalues[model_ols.pvalues < 0.05].drop('const', errors='ignore')
for var in sig_vars.index:
    coef = model_ols.params[var]
    direction = "양(+)" if coef > 0 else "음(-)"
    print(f"  - {var}: {direction} 영향 (계수={coef:.3f})")

print("\n[K-means 클러스터링 요약]")
print(f"- 최적 클러스터 수: {n_clusters}")
print(f"- Silhouette Score: {silhouette_score(X_cluster, kmeans_final.labels_):.3f}")

print("\n" + "=" * 60)
print("모델링 완료!")
print("=" * 60)

