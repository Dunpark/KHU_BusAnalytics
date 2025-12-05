"""
서울시 상권 데이터 종합 EDA 및 시각화
- AnalysisPlan.md 작성을 위한 근거 자료
- 시각화 결과 저장
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 저장 경로 설정
SAVE_PATH = './'
os.makedirs(SAVE_PATH, exist_ok=True)

# 데이터 로드
print("=" * 80)
print("📊 서울시 상권 데이터 종합 EDA 및 시각화")
print("=" * 80)

file_path = '../Merged_datasets/4개년_통합데이터_추정매출_상주인구_소득소비_길단위인구_점포_영역.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# 연도/분기 추출
df['기준_년_코드'] = df['기준_년분기_코드'] // 10
df['기준_분기_코드'] = df['기준_년분기_코드'] % 10

print(f"✅ 데이터 로딩 완료: {len(df):,}건, {df.shape[1]}개 컬럼")

# =============================================================================
# 1. 데이터 기본 구조 분석
# =============================================================================
print("\n" + "=" * 80)
print("1️⃣ 데이터 기본 구조")
print("=" * 80)

print(f"\n📌 기본 정보:")
print(f"  - 총 데이터 수: {len(df):,}건")
print(f"  - 컬럼 수: {df.shape[1]}개")
print(f"  - 기간: {df['기준_년_코드'].min()}년 ~ {df['기준_년_코드'].max()}년")
print(f"  - 고유 상권 수: {df['상권_코드'].nunique():,}개")
print(f"  - 고유 업종 수: {df['서비스_업종_코드'].nunique()}개")
print(f"  - 결측치: {df.isnull().sum().sum()}개")

# =============================================================================
# 2. 연도별/분기별 데이터 분포
# =============================================================================
print("\n" + "=" * 80)
print("2️⃣ 연도별/분기별 데이터 분포")
print("=" * 80)

# 연도-분기 조합별 데이터 수
yq_counts = df.groupby(['기준_년_코드', '기준_분기_코드']).size().unstack(fill_value=0)
print("\n[연도-분기 조합별 데이터 수]")
print(yq_counts)

# 시각화 1: 연도-분기별 데이터 분포 히트맵
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(yq_counts, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title('연도-분기별 데이터 분포', fontsize=14, fontweight='bold')
ax.set_xlabel('분기')
ax.set_ylabel('연도')
plt.tight_layout()
plt.savefig(f'{SAVE_PATH}fig1_year_quarter_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ 저장: fig1_year_quarter_distribution.png")

# =============================================================================
# 3. 상권 유형별 분포
# =============================================================================
print("\n" + "=" * 80)
print("3️⃣ 상권 유형별 분포")
print("=" * 80)

area_dist = df['상권_구분_코드_명'].value_counts()
print("\n[상권 유형별 데이터 수]")
for name, count in area_dist.items():
    pct = count / len(df) * 100
    print(f"  {name}: {count:,}건 ({pct:.1f}%)")

# 시각화 2: 상권 유형별 분포 파이차트
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 파이차트
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
axes[0].pie(area_dist.values, labels=area_dist.index, autopct='%1.1f%%', colors=colors, startangle=90)
axes[0].set_title('상권 유형별 데이터 비중', fontsize=12, fontweight='bold')

# 바차트
area_sales = df.groupby('상권_구분_코드_명')['당월_매출_금액'].mean() / 1e8
area_sales = area_sales.reindex(area_dist.index)
bars = axes[1].bar(area_sales.index, area_sales.values, color=colors)
axes[1].set_title('상권 유형별 평균 매출 (억원)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('평균 매출 (억원)')
for bar, val in zip(bars, area_sales.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(f'{SAVE_PATH}fig2_area_type_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ 저장: fig2_area_type_distribution.png")

# =============================================================================
# 4. 매출 분포 분석
# =============================================================================
print("\n" + "=" * 80)
print("4️⃣ 매출 분포 분석")
print("=" * 80)

print("\n[당월_매출_금액 기술통계]")
sales_stats = df['당월_매출_금액'].describe()
print(f"  평균: {sales_stats['mean']:,.0f}원")
print(f"  중앙값: {sales_stats['50%']:,.0f}원")
print(f"  표준편차: {sales_stats['std']:,.0f}원")
print(f"  최소: {sales_stats['min']:,.0f}원")
print(f"  최대: {sales_stats['max']:,.0f}원")
print(f"  왜도: {df['당월_매출_금액'].skew():.2f}")

# 로그 변환 효과
log_sales = np.log1p(df['당월_매출_금액'])
print(f"\n[로그 변환 후 왜도]")
print(f"  원본 왜도: {df['당월_매출_금액'].skew():.2f}")
print(f"  로그 변환 후: {log_sales.skew():.2f}")

# 시각화 3: 매출 분포 (원본 vs 로그)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 원본 분포
axes[0].hist(df['당월_매출_금액'] / 1e8, bins=100, color='#3498db', alpha=0.7, edgecolor='white')
axes[0].axvline(df['당월_매출_금액'].mean() / 1e8, color='red', linestyle='--', label=f'평균: {df["당월_매출_금액"].mean()/1e8:.1f}억')
axes[0].axvline(df['당월_매출_금액'].median() / 1e8, color='green', linestyle='--', label=f'중앙값: {df["당월_매출_금액"].median()/1e8:.1f}억')
axes[0].set_title(f'원본 매출 분포 (왜도: {df["당월_매출_금액"].skew():.1f})', fontsize=12, fontweight='bold')
axes[0].set_xlabel('당월 매출 (억원)')
axes[0].set_ylabel('빈도')
axes[0].set_xlim(0, 100)
axes[0].legend()

# 로그 변환 분포
axes[1].hist(log_sales, bins=100, color='#2ecc71', alpha=0.7, edgecolor='white')
axes[1].axvline(log_sales.mean(), color='red', linestyle='--', label=f'평균: {log_sales.mean():.1f}')
axes[1].axvline(log_sales.median(), color='green', linestyle='--', label=f'중앙값: {log_sales.median():.1f}')
axes[1].set_title(f'로그 변환 후 분포 (왜도: {log_sales.skew():.2f})', fontsize=12, fontweight='bold')
axes[1].set_xlabel('log(당월 매출)')
axes[1].set_ylabel('빈도')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{SAVE_PATH}fig3_sales_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ 저장: fig3_sales_distribution.png")

# =============================================================================
# 5. 연도별 매출 추세
# =============================================================================
print("\n" + "=" * 80)
print("5️⃣ 연도별 매출 추세")
print("=" * 80)

yearly_sales = df.groupby('기준_년_코드')['당월_매출_금액'].agg(['mean', 'median', 'sum']).reset_index()
yearly_sales.columns = ['연도', '평균매출', '중앙값매출', '총매출']

print("\n[연도별 평균 매출]")
for _, row in yearly_sales.iterrows():
    print(f"  {int(row['연도'])}년: {row['평균매출']/1e8:.2f}억원")

# 전년 대비 성장률
yearly_sales['성장률'] = yearly_sales['평균매출'].pct_change() * 100
print("\n[전년 대비 성장률]")
for _, row in yearly_sales.iterrows():
    if pd.notna(row['성장률']):
        print(f"  {int(row['연도'])}년: {row['성장률']:+.1f}%")

# 시각화 4: 연도별 매출 추세
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 평균 매출 추세
axes[0].plot(yearly_sales['연도'], yearly_sales['평균매출'] / 1e8, marker='o', linewidth=2, markersize=8, color='#3498db')
axes[0].fill_between(yearly_sales['연도'], yearly_sales['평균매출'] / 1e8, alpha=0.3, color='#3498db')
axes[0].set_title('연도별 평균 매출 추세', fontsize=12, fontweight='bold')
axes[0].set_xlabel('연도')
axes[0].set_ylabel('평균 매출 (억원)')
axes[0].set_xticks([2021, 2022, 2023, 2024])
for i, row in yearly_sales.iterrows():
    axes[0].annotate(f'{row["평균매출"]/1e8:.1f}억', (row['연도'], row['평균매출']/1e8), 
                     textcoords="offset points", xytext=(0,10), ha='center')

# 성장률 바차트
colors_growth = ['gray' if pd.isna(g) else ('#2ecc71' if g > 0 else '#e74c3c') for g in yearly_sales['성장률']]
bars = axes[1].bar(yearly_sales['연도'], yearly_sales['성장률'].fillna(0), color=colors_growth)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1].set_title('전년 대비 성장률 (%)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('연도')
axes[1].set_ylabel('성장률 (%)')
axes[1].set_xticks([2021, 2022, 2023, 2024])
for bar, val in zip(bars, yearly_sales['성장률'].fillna(0)):
    if val != 0:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{val:+.1f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(f'{SAVE_PATH}fig4_yearly_sales_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ 저장: fig4_yearly_sales_trend.png")

# =============================================================================
# 6. 업종별 매출 분석
# =============================================================================
print("\n" + "=" * 80)
print("6️⃣ 업종별 매출 분석")
print("=" * 80)

service_sales = df.groupby('서비스_업종_코드_명')['당월_매출_금액'].mean().sort_values(ascending=False)

print("\n[고매출 업종 TOP 10]")
for i, (name, val) in enumerate(service_sales.head(10).items(), 1):
    print(f"  {i:2d}. {name}: {val/1e8:.1f}억원")

print("\n[저매출 업종 TOP 10]")
for i, (name, val) in enumerate(service_sales.tail(10).items(), 1):
    print(f"  {i:2d}. {name}: {val/1e8:.2f}억원")

# 시각화 5: 업종별 매출 TOP 15
fig, ax = plt.subplots(figsize=(12, 8))
top15 = service_sales.head(15)
colors = plt.cm.Blues(np.linspace(0.4, 0.9, 15))[::-1]
bars = ax.barh(range(len(top15)), top15.values / 1e8, color=colors)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15.index)
ax.invert_yaxis()
ax.set_title('업종별 평균 매출 TOP 15', fontsize=14, fontweight='bold')
ax.set_xlabel('평균 매출 (억원)')
for bar, val in zip(bars, top15.values / 1e8):
    ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}억', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{SAVE_PATH}fig5_service_sales_top15.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ 저장: fig5_service_sales_top15.png")

# =============================================================================
# 7. 상관관계 분석
# =============================================================================
print("\n" + "=" * 80)
print("7️⃣ 주요 변수 상관관계 분석")
print("=" * 80)

corr_vars = ['당월_매출_금액', '점포_수', '총_상주인구_수', '총_유동인구_수', 
             '월_평균_소득_금액', '지출_총금액', '영역_면적', '유사_업종_점포_수']

corr_matrix = df[corr_vars].corr()

print("\n[당월_매출_금액과의 상관계수]")
sales_corr = corr_matrix['당월_매출_금액'].sort_values(ascending=False)
for var, corr in sales_corr.items():
    if var != '당월_매출_금액':
        print(f"  {var}: {corr:.3f}")

print("\n[높은 상관관계 쌍 (|r| > 0.7)]")
high_corr_found = False
for i in range(len(corr_vars)):
    for j in range(i+1, len(corr_vars)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.7:
            high_corr_found = True
            print(f"  {corr_vars[i]} ↔ {corr_vars[j]}: {corr_val:.3f}")
if not high_corr_found:
    print("  없음")

# 시각화 6: 상관관계 히트맵
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title('주요 변수 상관관계 히트맵', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{SAVE_PATH}fig6_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ 저장: fig6_correlation_heatmap.png")

# =============================================================================
# 8. 상권 성장률 분석
# =============================================================================
print("\n" + "=" * 80)
print("8️⃣ 상권 성장률 분석 (2021→2024)")
print("=" * 80)

# 상권-업종별 연도별 매출 집계
yearly_by_sangwon = df.groupby(['상권_코드', '서비스_업종_코드', '기준_년_코드'])['당월_매출_금액'].sum().reset_index()

sales_2021 = yearly_by_sangwon[yearly_by_sangwon['기준_년_코드'] == 2021][['상권_코드', '서비스_업종_코드', '당월_매출_금액']]
sales_2021.columns = ['상권_코드', '서비스_업종_코드', '매출_2021']

sales_2024 = yearly_by_sangwon[yearly_by_sangwon['기준_년_코드'] == 2024][['상권_코드', '서비스_업종_코드', '당월_매출_금액']]
sales_2024.columns = ['상권_코드', '서비스_업종_코드', '매출_2024']

growth_df = pd.merge(sales_2021, sales_2024, on=['상권_코드', '서비스_업종_코드'], how='inner')
growth_df['성장률'] = (growth_df['매출_2024'] - growth_df['매출_2021']) / growth_df['매출_2021'] * 100

# 성장률 분류
growth_df['성장_분류'] = pd.cut(growth_df['성장률'], 
                              bins=[-np.inf, -20, 20, np.inf],
                              labels=['쇠퇴(-20%이하)', '정체(-20~20%)', '성장(20%이상)'])

print("\n[상권-업종 조합 성장률 통계]")
print(f"  분석 대상: {len(growth_df):,}개")
print(f"  평균 성장률: {growth_df['성장률'].mean():.1f}%")
print(f"  중앙값 성장률: {growth_df['성장률'].median():.1f}%")

print("\n[성장 분류별 비율]")
growth_class = growth_df['성장_분류'].value_counts()
for cls, cnt in growth_class.items():
    pct = cnt / len(growth_df) * 100
    print(f"  {cls}: {cnt:,}개 ({pct:.1f}%)")

# 상권 단위 집계
sangwon_growth = growth_df.groupby('상권_코드')['성장률'].mean().reset_index()
sangwon_growth['상권_성장_분류'] = pd.cut(sangwon_growth['성장률'], 
                                        bins=[-np.inf, -20, 20, np.inf],
                                        labels=['쇠퇴', '정체', '성장'])

print("\n[상권 단위 성장 분류]")
sangwon_class = sangwon_growth['상권_성장_분류'].value_counts()
for cls, cnt in sangwon_class.items():
    pct = cnt / len(sangwon_growth) * 100
    print(f"  {cls}: {cnt:,}개 ({pct:.1f}%)")

# 시각화 7: 성장률 분포
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 성장률 히스토그램
axes[0].hist(growth_df['성장률'].clip(-100, 200), bins=50, color='#3498db', alpha=0.7, edgecolor='white')
axes[0].axvline(0, color='black', linestyle='-', linewidth=1)
axes[0].axvline(growth_df['성장률'].mean(), color='red', linestyle='--', label=f'평균: {growth_df["성장률"].mean():.1f}%')
axes[0].axvline(growth_df['성장률'].median(), color='green', linestyle='--', label=f'중앙값: {growth_df["성장률"].median():.1f}%')
axes[0].set_title('상권-업종별 성장률 분포 (2021→2024)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('성장률 (%)')
axes[0].set_ylabel('빈도')
axes[0].legend()

# 성장 분류 파이차트
colors = ['#e74c3c', '#f39c12', '#2ecc71']
axes[1].pie(growth_class.values, labels=growth_class.index, autopct='%1.1f%%', colors=colors, startangle=90)
axes[1].set_title('상권-업종 성장 분류', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{SAVE_PATH}fig7_growth_rate_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ 저장: fig7_growth_rate_distribution.png")

# =============================================================================
# 9. 업종×상권유형 교차 분석
# =============================================================================
print("\n" + "=" * 80)
print("9️⃣ 업종×상권유형 교차 분석")
print("=" * 80)

# 업종-상권유형별 평균 매출
cross_sales = df.groupby(['서비스_업종_코드_명', '상권_구분_코드_명'])['당월_매출_금액'].mean().unstack()

# 주요 업종의 상권유형별 매출
top_services = df['서비스_업종_코드_명'].value_counts().head(10).index.tolist()

print("\n[주요 업종별 최적 상권 유형]")
for service in top_services:
    if service in cross_sales.index:
        best_area = cross_sales.loc[service].idxmax()
        best_sales = cross_sales.loc[service].max()
        print(f"  {service}: {best_area} ({best_sales/1e8:.1f}억원)")

# 시각화 8: 업종×상권유형 히트맵
cross_sales_top = cross_sales.loc[top_services] / 1e8
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(cross_sales_top, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax)
ax.set_title('주요 업종 × 상권유형별 평균 매출 (억원)', fontsize=14, fontweight='bold')
ax.set_xlabel('상권 유형')
ax.set_ylabel('업종')
plt.tight_layout()
plt.savefig(f'{SAVE_PATH}fig8_service_area_cross.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ 저장: fig8_service_area_cross.png")

# =============================================================================
# 10. 자치구별 분포
# =============================================================================
print("\n" + "=" * 80)
print("🔟 자치구별 분포")
print("=" * 80)

gu_stats = df.groupby('자치구_코드_명').agg({
    '당월_매출_금액': 'mean',
    '상권_코드': 'nunique',
    '점포_수': 'mean'
}).round(0)
gu_stats.columns = ['평균매출', '상권수', '평균점포수']
gu_stats = gu_stats.sort_values('평균매출', ascending=False)

print("\n[자치구별 평균 매출 TOP 10]")
for i, (gu, row) in enumerate(gu_stats.head(10).iterrows(), 1):
    print(f"  {i:2d}. {gu}: {row['평균매출']/1e8:.1f}억원 (상권 {int(row['상권수'])}개)")

# 시각화 9: 자치구별 매출
fig, ax = plt.subplots(figsize=(14, 6))
gu_top15 = gu_stats.head(15)
colors = plt.cm.Reds(np.linspace(0.4, 0.9, 15))[::-1]
bars = ax.bar(range(len(gu_top15)), gu_top15['평균매출'] / 1e8, color=colors)
ax.set_xticks(range(len(gu_top15)))
ax.set_xticklabels(gu_top15.index, rotation=45, ha='right')
ax.set_title('자치구별 평균 매출 TOP 15', fontsize=14, fontweight='bold')
ax.set_ylabel('평균 매출 (억원)')
for bar, val in zip(bars, gu_top15['평균매출'] / 1e8):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{val:.1f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{SAVE_PATH}fig9_gu_sales.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ 저장: fig9_gu_sales.png")

# =============================================================================
# 11. 로그 변환 효과 비교
# =============================================================================
print("\n" + "=" * 80)
print("1️⃣1️⃣ 로그 변환 효과 비교")
print("=" * 80)

transform_vars = ['당월_매출_금액', '점포_수', '총_유동인구_수', '영역_면적']
skew_results = []

for var in transform_vars:
    original_skew = df[var].skew()
    log_skew = np.log1p(df[var]).skew()
    improvement = (abs(original_skew) - abs(log_skew)) / abs(original_skew) * 100
    skew_results.append({
        '변수': var,
        '원본_왜도': original_skew,
        '로그_왜도': log_skew,
        '개선율': improvement
    })
    print(f"  {var}: {original_skew:.2f} → {log_skew:.2f} ({improvement:.1f}% 개선)")

# 시각화 10: 로그 변환 효과
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, var in enumerate(transform_vars):
    row, col = idx // 2, idx % 2
    ax = axes[row, col]
    
    # 로그 변환된 데이터 히스토그램
    log_data = np.log1p(df[var])
    ax.hist(log_data, bins=50, color='#2ecc71', alpha=0.7, edgecolor='white')
    ax.set_title(f'{var}\n(원본 왜도: {df[var].skew():.1f} → 로그 왜도: {log_data.skew():.2f})', fontsize=10, fontweight='bold')
    ax.set_xlabel(f'log({var})')
    ax.set_ylabel('빈도')

plt.suptitle('로그 변환 후 분포', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{SAVE_PATH}fig10_log_transform_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ 저장: fig10_log_transform_effect.png")

# =============================================================================
# 완료
# =============================================================================
print("\n" + "=" * 80)
print("✅ 모든 분석 및 시각화 완료!")
print("=" * 80)
print(f"\n📁 저장된 파일 목록:")
for i in range(1, 11):
    print(f"  - fig{i}_*.png")

