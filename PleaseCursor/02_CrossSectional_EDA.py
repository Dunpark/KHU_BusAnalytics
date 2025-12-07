# -*- coding: utf-8 -*-
"""
02_CrossSectional_EDA.py
횡단면 EDA (Cross-Sectional Exploratory Data Analysis)
- 2024년 4분기 데이터 기준
- 상권 특성별 점포당 매출액 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("횡단면 EDA (2024년 4분기)")
print("=" * 60)

# ============================================================
# 1. 데이터 로드 및 전처리
# ============================================================
print("\n" + "=" * 60)
print("1. 데이터 로드 및 전처리")
print("=" * 60)

# 횡단면 데이터 로드
df_cross = pd.read_csv('df_cafe_20244.csv')
print(f"횡단면 데이터 로드: {df_cross.shape}")

# 시계열 특성 데이터 로드
df_ts_features = pd.read_csv('df_ts_features.csv')
print(f"시계열 특성 데이터 로드: {df_ts_features.shape}")

# DTW 클러스터 결과 로드 (있는 경우)
try:
    df_dtw = pd.read_csv('df_dtw_clusters.csv')
    print(f"DTW 클러스터 데이터 로드: {df_dtw.shape}")
    has_dtw = True
except FileNotFoundError:
    has_dtw = False
    print("DTW 클러스터 데이터 없음 - 규칙 기반 분류 사용")

# ============================================================
# 2. 파생변수 생성
# ============================================================
print("\n" + "=" * 60)
print("2. 파생변수 생성")
print("=" * 60)

# 점포당 매출액
df_cross['점포당_매출액'] = df_cross['당월_매출_금액'] / df_cross['유사_업종_점포_수']
df_cross['점포당_매출액'] = df_cross['점포당_매출액'].replace([np.inf, -np.inf], np.nan)

# 로그 변환
df_cross['log_점포당_매출액'] = np.log1p(df_cross['점포당_매출액'])
df_cross['log_당월_매출_금액'] = np.log1p(df_cross['당월_매출_금액'])

# 인구 관련 비율
df_cross['유동인구_대비_상주인구'] = df_cross['총_상주인구_수'] / (df_cross['총_유동인구_수'] + 1)
df_cross['남성_비율'] = df_cross['남성_상주인구_수'] / (df_cross['총_상주인구_수'] + 1)
df_cross['청년_비율'] = (df_cross['연령대_20_상주인구_수'] + df_cross['연령대_30_상주인구_수']) / (df_cross['총_상주인구_수'] + 1)

# 경쟁 강도
df_cross['경쟁강도'] = df_cross['유사_업종_점포_수'] / (df_cross['영역_면적'] / 10000 + 1)  # 만㎡당 점포수

# 소득 수준 (소득 구간 코드 활용)
df_cross['고소득_여부'] = (df_cross['소득_구간_코드'] >= 7).astype(int)

# 프랜차이즈 비율
df_cross['프랜차이즈_비율'] = df_cross['프랜차이즈_점포_수'] / (df_cross['유사_업종_점포_수'] + 1)

print(f"파생변수 생성 완료")
print(f"생성된 변수: 점포당_매출액, log_점포당_매출액, 유동인구_대비_상주인구, 남성_비율, 청년_비율, 경쟁강도, 프랜차이즈_비율")

# ============================================================
# 3. 시계열 특성과 병합
# ============================================================
print("\n" + "=" * 60)
print("3. 시계열 특성과 병합")
print("=" * 60)

# 시계열 특성 병합
ts_cols = ['상권_코드', 'sps_mean', 'sps_growth', 'sps_cagr', 'sps_cv', 
           'store_growth', 'decline_ratio', '상권유형']

if has_dtw:
    # DTW 결과 우선 사용
    df_merged = df_cross.merge(
        df_dtw[['상권_코드', 'DTW_Cluster', '상권유형']],
        on='상권_코드',
        how='left'
    )
    # 시계열 특성도 추가
    df_merged = df_merged.merge(
        df_ts_features[['상권_코드', 'sps_mean', 'sps_growth', 'sps_cagr', 'sps_cv', 
                        'store_growth', 'decline_ratio']],
        on='상권_코드',
        how='left'
    )
else:
    df_merged = df_cross.merge(
        df_ts_features[ts_cols],
        on='상권_코드',
        how='left'
    )

print(f"병합 완료: {df_merged.shape}")
print(f"상권유형 분포:\n{df_merged['상권유형'].value_counts()}")

# ============================================================
# 4. 상관관계 분석
# ============================================================
print("\n" + "=" * 60)
print("4. 상관관계 분석")
print("=" * 60)

# 분석 대상 변수
analysis_vars = ['점포당_매출액', '당월_매출_금액', '유사_업종_점포_수', 
                 '총_유동인구_수', '총_상주인구_수', '월_평균_소득_금액',
                 '경쟁강도', '프랜차이즈_비율', '청년_비율']

# 상관관계 계산
corr_matrix = df_merged[analysis_vars].corr()

# 점포당 매출액과의 상관관계
print("\n점포당 매출액과의 상관관계:")
sps_corr = corr_matrix['점포당_매출액'].drop('점포당_매출액').sort_values(ascending=False)
for var, corr in sps_corr.items():
    print(f"  {var}: {corr:.3f}")

# 시각화: 상관관계 히트맵
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 히트맵
ax1 = axes[0]
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax1, square=True, linewidths=0.5)
ax1.set_title('변수 간 상관관계 히트맵', fontsize=14, fontweight='bold')

# 점포당 매출액 상관관계 바차트
ax2 = axes[1]
colors = ['#2E86AB' if x > 0 else '#E94F37' for x in sps_corr.values]
bars = ax2.barh(sps_corr.index, sps_corr.values, color=colors, alpha=0.8)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_title('점포당 매출액과의 상관관계', fontsize=14, fontweight='bold')
ax2.set_xlabel('상관계수')

plt.tight_layout()
plt.savefig('cross_01_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[저장] cross_01_correlation.png")

# ============================================================
# 5. 상권유형별 비교 분석
# ============================================================
print("\n" + "=" * 60)
print("5. 상권유형별 비교 분석")
print("=" * 60)

if df_merged['상권유형'].notna().sum() > 0:
    type_order = ['성장상권', '기대상권', '전통강호상권', '정체상권', '쇠퇴상권']
    colors_type = {'성장상권': '#2E86AB', '기대상권': '#A23B72', '전통강호상권': '#F18F01',
                   '정체상권': '#C73E1D', '쇠퇴상권': '#3B1F2B'}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 5-1. 점포당 매출액 분포
    ax1 = axes[0, 0]
    for t in type_order:
        data = df_merged[df_merged['상권유형'] == t]['점포당_매출액'] / 1e6
        data = data.dropna()
        if len(data) > 0:
            ax1.hist(data, bins=30, alpha=0.5, label=t, color=colors_type[t])
    ax1.set_title('상권유형별 점포당 매출액 분포', fontsize=12, fontweight='bold')
    ax1.set_xlabel('점포당 매출액 (백만원)')
    ax1.set_ylabel('빈도')
    ax1.legend()
    ax1.set_xlim(0, 150)
    
    # 5-2. 박스플롯
    ax2 = axes[0, 1]
    boxplot_data = []
    labels_bp = []
    for t in type_order:
        data = df_merged[df_merged['상권유형'] == t]['점포당_매출액'] / 1e6
        data = data.dropna()
        if len(data) > 0:
            boxplot_data.append(data)
            labels_bp.append(t)
    
    bp = ax2.boxplot(boxplot_data, labels=labels_bp, patch_artist=True)
    for patch, t in zip(bp['boxes'], labels_bp):
        patch.set_facecolor(colors_type[t])
        patch.set_alpha(0.7)
    ax2.set_title('상권유형별 점포당 매출액 박스플롯', fontsize=12, fontweight='bold')
    ax2.set_ylabel('점포당 매출액 (백만원)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim(0, 150)
    
    # 5-3. 유동인구 비교
    ax3 = axes[1, 0]
    floating_by_type = df_merged.groupby('상권유형')['총_유동인구_수'].mean().reindex(type_order) / 1e6
    bars = ax3.bar(type_order, floating_by_type, 
                   color=[colors_type[t] for t in type_order], alpha=0.8)
    ax3.set_title('상권유형별 평균 유동인구', fontsize=12, fontweight='bold')
    ax3.set_ylabel('유동인구 (백만명)')
    ax3.tick_params(axis='x', rotation=45)
    
    # 5-4. 경쟁강도 비교
    ax4 = axes[1, 1]
    competition_by_type = df_merged.groupby('상권유형')['경쟁강도'].mean().reindex(type_order)
    bars = ax4.bar(type_order, competition_by_type,
                   color=[colors_type[t] for t in type_order], alpha=0.8)
    ax4.set_title('상권유형별 평균 경쟁강도', fontsize=12, fontweight='bold')
    ax4.set_ylabel('경쟁강도 (만㎡당 점포수)')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('cross_02_type_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[저장] cross_02_type_comparison.png")
    
    # 상권유형별 통계
    print("\n상권유형별 주요 지표:")
    type_stats = df_merged.groupby('상권유형').agg({
        '점포당_매출액': ['mean', 'median', 'std'],
        '총_유동인구_수': 'mean',
        '경쟁강도': 'mean',
        '프랜차이즈_비율': 'mean'
    }).round(2)
    print(type_stats)

# ============================================================
# 6. 자치구별 분석
# ============================================================
print("\n" + "=" * 60)
print("6. 자치구별 분석")
print("=" * 60)

# 자치구별 점포당 매출액
gu_stats = df_merged.groupby('자치구_코드_명').agg({
    '점포당_매출액': ['mean', 'median', 'count'],
    '당월_매출_금액': 'sum',
    '유사_업종_점포_수': 'sum'
}).round(0)

gu_stats.columns = ['평균_점포당매출', '중앙값_점포당매출', '상권수', '총매출', '총점포수']
gu_stats = gu_stats.sort_values('평균_점포당매출', ascending=False)

print("\n자치구별 점포당 매출액 순위 (상위 10):")
print((gu_stats['평균_점포당매출'] / 1e6).head(10).round(1))

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# 6-1. 자치구별 점포당 매출액
ax1 = axes[0]
top_gu = (gu_stats['평균_점포당매출'] / 1e6).head(15)
colors_gu = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(top_gu)))
bars = ax1.barh(top_gu.index, top_gu.values, color=colors_gu, alpha=0.8)
ax1.set_title('자치구별 평균 점포당 매출액 (상위 15)', fontsize=12, fontweight='bold')
ax1.set_xlabel('점포당 매출액 (백만원)')
ax1.invert_yaxis()

# 6-2. 자치구별 상권 수
ax2 = axes[1]
gu_count = gu_stats['상권수'].sort_values(ascending=False).head(15)
bars = ax2.barh(gu_count.index, gu_count.values, color='#2E86AB', alpha=0.8)
ax2.set_title('자치구별 상권 수 (상위 15)', fontsize=12, fontweight='bold')
ax2.set_xlabel('상권 수')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('cross_03_gu_ranking.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[저장] cross_03_gu_ranking.png")

# ============================================================
# 7. 상권 구분별 분석
# ============================================================
print("\n" + "=" * 60)
print("7. 상권 구분별 분석")
print("=" * 60)

# 상권 구분 코드별 분석
area_type_stats = df_merged.groupby('상권_구분_코드_명').agg({
    '점포당_매출액': ['mean', 'median', 'std', 'count'],
    '당월_매출_금액': 'sum',
    '총_유동인구_수': 'mean'
}).round(0)

print("\n상권 구분별 점포당 매출액:")
print(area_type_stats)

# 시각화
fig, ax = plt.subplots(figsize=(10, 6))

area_sps = df_merged.groupby('상권_구분_코드_명')['점포당_매출액'].mean() / 1e6
area_sps = area_sps.sort_values(ascending=False)

colors_area = plt.cm.viridis(np.linspace(0, 1, len(area_sps)))
bars = ax.bar(area_sps.index, area_sps.values, color=colors_area, alpha=0.8)
ax.set_title('상권 구분별 평균 점포당 매출액', fontsize=14, fontweight='bold')
ax.set_xlabel('상권 구분')
ax.set_ylabel('점포당 매출액 (백만원)')
ax.tick_params(axis='x', rotation=45)

for bar, val in zip(bars, area_sps.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('cross_04_area_type.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[저장] cross_04_area_type.png")

# ============================================================
# 8. 주요 변수 분포 분석
# ============================================================
print("\n" + "=" * 60)
print("8. 주요 변수 분포 분석")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

dist_vars = ['점포당_매출액', '유사_업종_점포_수', '총_유동인구_수', 
             '월_평균_소득_금액', '경쟁강도', '프랜차이즈_비율']
titles = ['점포당 매출액', '유사업종 점포수', '유동인구', 
          '월평균 소득', '경쟁강도', '프랜차이즈 비율']

for i, (var, title) in enumerate(zip(dist_vars, titles)):
    ax = axes[i]
    data = df_merged[var].dropna()
    
    # 이상치 제거 (상위/하위 1%)
    lower = data.quantile(0.01)
    upper = data.quantile(0.99)
    data_clean = data[(data >= lower) & (data <= upper)]
    
    ax.hist(data_clean, bins=50, color='#2E86AB', alpha=0.7, edgecolor='white')
    ax.axvline(data_clean.mean(), color='red', linestyle='--', label=f'평균: {data_clean.mean():.1f}')
    ax.axvline(data_clean.median(), color='green', linestyle='--', label=f'중앙값: {data_clean.median():.1f}')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cross_05_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[저장] cross_05_distributions.png")

# ============================================================
# 9. 결과 저장
# ============================================================
print("\n" + "=" * 60)
print("9. 결과 저장")
print("=" * 60)

# 모델링용 데이터 저장
modeling_cols = ['상권_코드', '상권_코드_명', '자치구_코드_명', '상권_구분_코드_명',
                 '점포당_매출액', 'log_점포당_매출액', '당월_매출_금액', '유사_업종_점포_수',
                 '총_유동인구_수', '총_상주인구_수', '월_평균_소득_금액',
                 '경쟁강도', '프랜차이즈_비율', '청년_비율', '유동인구_대비_상주인구',
                 '개업_율', '폐업_률', '영역_면적']

if '상권유형' in df_merged.columns:
    modeling_cols.append('상권유형')
if 'sps_growth' in df_merged.columns:
    modeling_cols.extend(['sps_growth', 'sps_cv', 'store_growth'])

# 존재하는 컬럼만 선택
existing_cols = [col for col in modeling_cols if col in df_merged.columns]
df_modeling = df_merged[existing_cols].copy()

df_modeling.to_csv('df_modeling.csv', index=False, encoding='utf-8-sig')
print("\n[저장] df_modeling.csv")

# 요약 통계 저장
summary = df_modeling.describe().round(2)
summary.to_csv('df_modeling_summary.csv', encoding='utf-8-sig')
print("[저장] df_modeling_summary.csv")

# ============================================================
# 10. 주요 발견 사항
# ============================================================
print("\n" + "=" * 60)
print("10. 주요 발견 사항")
print("=" * 60)

print("\n[데이터 개요]")
print(f"- 분석 상권 수: {len(df_merged)}")
print(f"- 분석 기준: 2024년 4분기")

print("\n[점포당 매출액 기초통계]")
sps = df_merged['점포당_매출액'].dropna()
print(f"- 평균: {sps.mean()/1e6:.1f}백만원")
print(f"- 중앙값: {sps.median()/1e6:.1f}백만원")
print(f"- 표준편차: {sps.std()/1e6:.1f}백만원")
print(f"- 최소: {sps.min()/1e6:.1f}백만원")
print(f"- 최대: {sps.max()/1e6:.1f}백만원")

print("\n[주요 상관관계]")
for var, corr in sps_corr.head(5).items():
    direction = "양의" if corr > 0 else "음의"
    print(f"- {var}: {direction} 상관관계 (r={corr:.3f})")

print("\n" + "=" * 60)
print("횡단면 EDA 완료!")
print("=" * 60)

