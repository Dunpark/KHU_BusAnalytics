# -*- coding: utf-8 -*-
"""
01_TimeSeries_EDA.py
시계열 EDA 및 특성 추출
- 점포당 매출액(SPS) 기반 분석
- 선행연구 방법론 적용
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
print("시계열 EDA 및 특성 추출")
print("=" * 60)

# 데이터 로드
df_ts = pd.read_csv('df_cafe_timeSeries.csv')
print(f"\n데이터 로드 완료: {df_ts.shape}")

# ============================================================
# 1. 점포당 매출액 계산
# ============================================================
print("\n" + "=" * 60)
print("1. 점포당 매출액(SPS) 계산")
print("=" * 60)

# 점포당 매출액 = 당월 매출 금액 / 유사 업종 점포 수
df_ts['점포당_매출액'] = df_ts['당월_매출_금액'] / df_ts['유사_업종_점포_수']
df_ts['점포당_매출액'] = df_ts['점포당_매출액'].replace([np.inf, -np.inf], np.nan)

print(f"점포당 매출액 기초통계:")
print(df_ts['점포당_매출액'].describe())

# ============================================================
# 2. 시계열 추세 분석
# ============================================================
print("\n" + "=" * 60)
print("2. 시계열 추세 분석")
print("=" * 60)

# 분기별 집계
quarterly_stats = df_ts.groupby('기준_년분기_코드').agg({
    '점포당_매출액': ['mean', 'median', 'std'],
    '당월_매출_금액': ['mean', 'sum'],
    '유사_업종_점포_수': ['mean', 'sum']
}).round(0)

print("\n분기별 점포당 매출액 추이:")
print(quarterly_stats)

# 시각화: 점포당 매출액 추세
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 2-1. 점포당 매출액 추세
ax1 = axes[0, 0]
quarterly_sps = df_ts.groupby('기준_년분기_코드')['점포당_매출액'].mean() / 1e6
quarterly_sps.plot(ax=ax1, marker='o', linewidth=2, color='#2E86AB')
ax1.set_title('분기별 평균 점포당 매출액 추이', fontsize=12, fontweight='bold')
ax1.set_xlabel('분기')
ax1.set_ylabel('점포당 매출액 (백만원)')
ax1.grid(True, alpha=0.3)

# 2-2. 총매출 vs 점포당 매출 비교 (정규화)
ax2 = axes[0, 1]
norm_sps = (quarterly_sps - quarterly_sps.min()) / (quarterly_sps.max() - quarterly_sps.min())
quarterly_sales = df_ts.groupby('기준_년분기_코드')['당월_매출_금액'].mean() / 1e6
norm_sales = (quarterly_sales - quarterly_sales.min()) / (quarterly_sales.max() - quarterly_sales.min())
ax2.plot(norm_sps.index, norm_sps.values, marker='o', label='점포당 매출액', color='#2E86AB')
ax2.plot(norm_sales.index, norm_sales.values, marker='s', label='총매출액', color='#E94F37')
ax2.set_title('점포당 매출액 vs 총매출액 (정규화)', fontsize=12, fontweight='bold')
ax2.set_xlabel('분기')
ax2.set_ylabel('정규화된 값')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 2-3. 점포수 추이
ax3 = axes[1, 0]
quarterly_stores = df_ts.groupby('기준_년분기_코드')['유사_업종_점포_수'].mean()
quarterly_stores.plot(ax=ax3, marker='o', linewidth=2, color='#F77F00')
ax3.set_title('분기별 평균 점포수 추이', fontsize=12, fontweight='bold')
ax3.set_xlabel('분기')
ax3.set_ylabel('점포수')
ax3.grid(True, alpha=0.3)

# 2-4. 점포당 매출액 분포 변화
ax4 = axes[1, 1]
years = [20211, 20221, 20231, 20244]
colors = ['#264653', '#2A9D8F', '#E9C46A', '#E76F51']
for i, year in enumerate(years):
    data = df_ts[df_ts['기준_년분기_코드'] == year]['점포당_매출액'] / 1e6
    data = data.dropna()
    if len(data) > 0:
        ax4.hist(data, bins=30, alpha=0.5, label=str(year), color=colors[i])
ax4.set_title('점포당 매출액 분포 변화', fontsize=12, fontweight='bold')
ax4.set_xlabel('점포당 매출액 (백만원)')
ax4.set_ylabel('빈도')
ax4.legend()
ax4.set_xlim(0, 100)

plt.tight_layout()
plt.savefig('ts_01_sps_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[저장] ts_01_sps_trend.png")

# ============================================================
# 3. 총매출 vs 점포당매출 vs 점포수 비교
# ============================================================
print("\n" + "=" * 60)
print("3. 총매출 vs 점포당매출 vs 점포수 비교")
print("=" * 60)

fig, ax = plt.subplots(figsize=(12, 6))

# 정규화된 값으로 비교
quarterly_data = df_ts.groupby('기준_년분기_코드').agg({
    '점포당_매출액': 'mean',
    '당월_매출_금액': 'mean',
    '유사_업종_점포_수': 'mean'
})

for col in quarterly_data.columns:
    quarterly_data[f'{col}_norm'] = (quarterly_data[col] - quarterly_data[col].min()) / \
                                     (quarterly_data[col].max() - quarterly_data[col].min())

ax.plot(quarterly_data.index, quarterly_data['점포당_매출액_norm'], 
        marker='o', label='점포당 매출액', linewidth=2, color='#2E86AB')
ax.plot(quarterly_data.index, quarterly_data['당월_매출_금액_norm'], 
        marker='s', label='총매출액', linewidth=2, color='#E94F37')
ax.plot(quarterly_data.index, quarterly_data['유사_업종_점포_수_norm'], 
        marker='^', label='점포수', linewidth=2, color='#F77F00')

ax.set_title('총매출 vs 점포당매출 vs 점포수 추이 비교 (정규화)', fontsize=14, fontweight='bold')
ax.set_xlabel('분기', fontsize=12)
ax.set_ylabel('정규화된 값', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ts_02_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[저장] ts_02_comparison.png")

# ============================================================
# 4. 상권별 점포당 매출액 박스플롯
# ============================================================
print("\n" + "=" * 60)
print("4. 상권별 점포당 매출액 분포")
print("=" * 60)

fig, ax = plt.subplots(figsize=(14, 6))

# 분기별 박스플롯
boxplot_data = []
labels = []
for q in sorted(df_ts['기준_년분기_코드'].unique()):
    data = df_ts[df_ts['기준_년분기_코드'] == q]['점포당_매출액'] / 1e6
    data = data.dropna()
    boxplot_data.append(data)
    labels.append(str(q))

bp = ax.boxplot(boxplot_data, labels=labels, patch_artist=True)
colors_bp = plt.cm.viridis(np.linspace(0, 1, len(boxplot_data)))
for patch, color in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_title('분기별 점포당 매출액 분포', fontsize=14, fontweight='bold')
ax.set_xlabel('분기', fontsize=12)
ax.set_ylabel('점포당 매출액 (백만원)', fontsize=12)
ax.set_ylim(0, 150)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('ts_03_sps_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[저장] ts_03_sps_boxplot.png")

# ============================================================
# 5. 상권별 시계열 특성 추출
# ============================================================
print("\n" + "=" * 60)
print("5. 상권별 시계열 특성 추출")
print("=" * 60)

def extract_sps_features(group):
    """상권별 점포당 매출액 시계열 특성 추출"""
    sorted_group = group.sort_values('기준_년분기_코드')
    sps = sorted_group['점포당_매출액'].values
    stores = sorted_group['유사_업종_점포_수'].values
    total_sales = sorted_group['당월_매출_금액'].values
    quarters = sorted_group['기준_년분기_코드'].values
    
    features = {}
    
    # =====================================================
    # 기본 통계 (점포당 매출액)
    # =====================================================
    # sps_mean: 20211~20244 동안 해당 상권의 점포당 매출액 평균
    # 수식: (점포당_매출액_20211 + 점포당_매출액_20212 + ... + 점포당_매출액_20244) / 분기수
    features['sps_mean'] = np.mean(sps)
    
    # sps_std: 점포당 매출액의 표준편차 (변동성)
    features['sps_std'] = np.std(sps)
    
    # sps_cv: 변동계수 (Coefficient of Variation) = 표준편차 / 평균
    # 규모와 무관한 안정성 비교 가능
    features['sps_cv'] = features['sps_std'] / features['sps_mean'] if features['sps_mean'] > 0 else np.nan
    
    features['sps_min'] = np.min(sps)
    features['sps_max'] = np.max(sps)
    features['sps_first'] = sps[0]  # 2021Q1
    features['sps_last'] = sps[-1]  # 2024Q4
    
    # =====================================================
    # 성장률 지표 (선행연구 핵심)
    # =====================================================
    # sps_growth: 4년 성장률
    # 수식: (점포당_매출액_20244 - 점포당_매출액_20211) / 점포당_매출액_20211
    if features['sps_first'] > 0:
        features['sps_growth'] = (features['sps_last'] - features['sps_first']) / features['sps_first']
        # sps_cagr: 연평균 성장률 (CAGR)
        # 수식: (점포당_매출액_20244 / 점포당_매출액_20211)^(1/4) - 1
        features['sps_cagr'] = (features['sps_last'] / features['sps_first']) ** (1 / 4) - 1
    else:
        features['sps_growth'] = np.nan
        features['sps_cagr'] = np.nan
    
    # 최근 1년 성장률 (YoY Growth)
    if len(sps) >= 8:
        recent_avg = np.mean(sps[-4:])  # 최근 4분기 평균
        prev_avg = np.mean(sps[-8:-4])  # 이전 4분기 평균
        features['sps_yoy_growth'] = (recent_avg - prev_avg) / prev_avg if prev_avg > 0 else np.nan
    else:
        features['sps_yoy_growth'] = np.nan
    
    # =====================================================
    # 추세 분석 (선형회귀)
    # =====================================================
    if len(sps) >= 4:
        x = np.arange(len(sps))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, sps)
        features['sps_trend_slope'] = slope
        features['sps_trend_r2'] = r_value ** 2
        features['sps_trend_pvalue'] = p_value
    else:
        features['sps_trend_slope'] = np.nan
        features['sps_trend_r2'] = np.nan
        features['sps_trend_pvalue'] = np.nan
    
    # =====================================================
    # 점포수 변화 (선행연구: 점포수 변동이 점포당 매출에 미치는 영향)
    # =====================================================
    features['store_first'] = stores[0]  # 2021Q1 유사 업종 점포 수
    features['store_last'] = stores[-1]  # 2024Q4 유사 업종 점포 수
    
    # store_growth: 점포수 변화율
    # 수식: (유사_업종_점포_수_20244 - 유사_업종_점포_수_20211) / 유사_업종_점포_수_20211
    if features['store_first'] > 0:
        features['store_growth'] = (features['store_last'] - features['store_first']) / features['store_first']
    else:
        features['store_growth'] = np.nan
    features['store_mean'] = np.mean(stores)
    
    # 총매출 (비교용)
    features['sales_mean'] = np.mean(total_sales)
    features['sales_growth'] = (total_sales[-1] - total_sales[0]) / total_sales[0] if total_sales[0] > 0 else np.nan
    
    # =====================================================
    # 계절성 지표
    # =====================================================
    quarter_nums = quarters % 10
    for q in [1, 2, 3, 4]:
        q_mask = quarter_nums == q
        if np.sum(q_mask) > 0:
            features[f'sps_q{q}_ratio'] = np.mean(sps[q_mask]) / features['sps_mean'] if features['sps_mean'] > 0 else np.nan
        else:
            features[f'sps_q{q}_ratio'] = np.nan
    
    # =====================================================
    # 하락 분기 분석
    # =====================================================
    if len(sps) >= 2:
        diff = np.diff(sps)
        features['decline_quarters'] = np.sum(diff < 0)
        features['decline_ratio'] = features['decline_quarters'] / (len(sps) - 1)
    else:
        features['decline_quarters'] = np.nan
        features['decline_ratio'] = np.nan
    
    # 데이터 충실도
    features['data_quarters'] = len(sps)
    
    return pd.Series(features)

# 특성 추출 실행
print("\n특성 추출 중...")
df_features = df_ts.groupby('상권_코드').apply(extract_sps_features).reset_index()
print(f"특성 추출 완료: {df_features.shape}")

# ============================================================
# 6. 상권 유형 분류 (규칙 기반 - DTW 클러스터링 이전 참고용)
# ============================================================
print("\n" + "=" * 60)
print("6. 상권 유형 분류 (규칙 기반)")
print("=" * 60)

# 성장/하락 분류
growth_median = df_features['sps_growth'].median()
df_features['trend_class'] = df_features['sps_growth'].apply(
    lambda x: '상승' if x > growth_median else ('하락' if x < -0.1 else '정체')
)

# 규모 분류 (점포당 매출 평균 기준)
sps_33 = df_features['sps_mean'].quantile(0.33)
sps_66 = df_features['sps_mean'].quantile(0.66)
df_features['scale_class'] = df_features['sps_mean'].apply(
    lambda x: '大' if x >= sps_66 else ('中' if x >= sps_33 else '小')
)

# 5가지 상권 유형 분류 (규칙 기반)
def classify_commercial_area(row):
    """
    상권 유형 분류 (규칙 기반 - 참고용)
    실제 분석에서는 DTW 클러스터링 결과 사용 권장
    """
    trend = row['trend_class']
    scale = row['scale_class']
    
    if trend == '상승' and scale == '大':
        return '성장상권'
    elif trend == '상승' and scale in ['中', '小']:
        return '기대상권'
    elif trend == '정체' and scale == '大':
        return '전통강호상권'
    elif trend == '정체' and scale in ['中', '小']:
        return '정체상권'
    else:  # trend == '하락'
        return '쇠퇴상권'

df_features['상권유형'] = df_features.apply(classify_commercial_area, axis=1)

print("\n상권 유형별 분포:")
print(df_features['상권유형'].value_counts())

# ============================================================
# 7. 샘플 상권 시계열 시각화
# ============================================================
print("\n" + "=" * 60)
print("7. 샘플 상권 시계열 시각화")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# 각 유형별 샘플 상권
types = ['성장상권', '기대상권', '전통강호상권', '정체상권', '쇠퇴상권']
colors_type = {'성장상권': '#2E86AB', '기대상권': '#A23B72', '전통강호상권': '#F18F01',
               '정체상권': '#C73E1D', '쇠퇴상권': '#3B1F2B'}

for i, area_type in enumerate(types):
    ax = axes[i]
    sample_codes = df_features[df_features['상권유형'] == area_type]['상권_코드'].head(3)
    
    for code in sample_codes:
        ts_data = df_ts[df_ts['상권_코드'] == code].sort_values('기준_년분기_코드')
        ax.plot(range(len(ts_data)), ts_data['점포당_매출액'] / 1e6, 
                marker='o', alpha=0.7, label=f'상권 {code}')
    
    ax.set_title(f'{area_type}', fontsize=12, fontweight='bold', color=colors_type[area_type])
    ax.set_xlabel('분기')
    ax.set_ylabel('점포당 매출액 (백만원)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# 마지막 subplot에 유형별 평균 추세
ax = axes[5]
for area_type in types:
    codes = df_features[df_features['상권유형'] == area_type]['상권_코드']
    type_data = df_ts[df_ts['상권_코드'].isin(codes)].groupby('기준_년분기_코드')['점포당_매출액'].mean() / 1e6
    ax.plot(range(len(type_data)), type_data.values, marker='o', 
            label=area_type, color=colors_type[area_type], linewidth=2)

ax.set_title('상권유형별 평균 추세', fontsize=12, fontweight='bold')
ax.set_xlabel('분기')
ax.set_ylabel('점포당 매출액 (백만원)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ts_04_sample_trends.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[저장] ts_04_sample_trends.png")

# ============================================================
# 8. 상권 유형별 특성 비교
# ============================================================
print("\n" + "=" * 60)
print("8. 상권 유형별 특성 비교")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 8-1. 점포당 매출액 평균
ax1 = axes[0, 0]
type_order = ['성장상권', '기대상권', '전통강호상권', '정체상권', '쇠퇴상권']
sps_by_type = df_features.groupby('상권유형')['sps_mean'].mean().reindex(type_order) / 1e6
colors_bar = [colors_type[t] for t in type_order]
bars = ax1.bar(type_order, sps_by_type, color=colors_bar, alpha=0.8)
ax1.set_title('상권유형별 평균 점포당 매출액', fontsize=12, fontweight='bold')
ax1.set_ylabel('점포당 매출액 (백만원)')
ax1.tick_params(axis='x', rotation=45)

# 8-2. 성장률
ax2 = axes[0, 1]
growth_by_type = df_features.groupby('상권유형')['sps_growth'].mean().reindex(type_order) * 100
ax2.bar(type_order, growth_by_type, color=colors_bar, alpha=0.8)
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax2.set_title('상권유형별 평균 성장률', fontsize=12, fontweight='bold')
ax2.set_ylabel('성장률 (%)')
ax2.tick_params(axis='x', rotation=45)

# 8-3. 변동계수
ax3 = axes[1, 0]
cv_by_type = df_features.groupby('상권유형')['sps_cv'].mean().reindex(type_order)
ax3.bar(type_order, cv_by_type, color=colors_bar, alpha=0.8)
ax3.set_title('상권유형별 평균 변동계수 (안정성)', fontsize=12, fontweight='bold')
ax3.set_ylabel('변동계수')
ax3.tick_params(axis='x', rotation=45)

# 8-4. 점포수 변화율
ax4 = axes[1, 1]
store_growth_by_type = df_features.groupby('상권유형')['store_growth'].mean().reindex(type_order) * 100
ax4.bar(type_order, store_growth_by_type, color=colors_bar, alpha=0.8)
ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax4.set_title('상권유형별 평균 점포수 변화율', fontsize=12, fontweight='bold')
ax4.set_ylabel('점포수 변화율 (%)')
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('ts_05_commercial_types.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[저장] ts_05_commercial_types.png")

# ============================================================
# 9. 결과 저장
# ============================================================
print("\n" + "=" * 60)
print("9. 결과 저장")
print("=" * 60)

# 전체 특성 저장
df_features.to_csv('df_ts_features.csv', index=False, encoding='utf-8-sig')
print("\n[저장] df_ts_features.csv")

# 요약 통계 저장
summary_stats = df_features.groupby('상권유형').agg({
    'sps_mean': ['mean', 'std', 'count'],
    'sps_growth': ['mean', 'std'],
    'sps_cv': ['mean', 'std'],
    'store_growth': ['mean', 'std']
}).round(2)
summary_stats.to_csv('df_ts_features_summary.csv', encoding='utf-8-sig')
print("[저장] df_ts_features_summary.csv")

# ============================================================
# 10. 주요 발견 사항 출력
# ============================================================
print("\n" + "=" * 60)
print("10. 주요 발견 사항")
print("=" * 60)

print("\n[시계열 특성 요약]")
print(f"- 전체 상권 수: {len(df_features)}")
print(f"- 분석 기간: 2021Q1 ~ 2024Q4 (16개 분기)")

print("\n[특성별 기초통계]")
key_features = ['sps_mean', 'sps_growth', 'sps_cagr', 'sps_cv', 'store_growth']
for feat in key_features:
    mean_val = df_features[feat].mean()
    std_val = df_features[feat].std()
    if 'growth' in feat or 'cagr' in feat or 'cv' in feat:
        print(f"- {feat}: 평균 {mean_val*100:.1f}%, 표준편차 {std_val*100:.1f}%")
    else:
        print(f"- {feat}: 평균 {mean_val/1e6:.1f}백만원, 표준편차 {std_val/1e6:.1f}백만원")

print("\n[상권유형별 분포]")
type_dist = df_features['상권유형'].value_counts()
for t, c in type_dist.items():
    print(f"- {t}: {c}개 ({c/len(df_features)*100:.1f}%)")

print("\n" + "=" * 60)
print("시계열 EDA 완료!")
print("=" * 60)

