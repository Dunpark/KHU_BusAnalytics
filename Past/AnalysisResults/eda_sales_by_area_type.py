"""
상권 유형별 매출 분포 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
print("=" * 80)
print("📊 상권 유형별 매출 분포 분석")
print("=" * 80)

file_path = 'KHU_BusAnalytics/Merged_datasets/4개년_통합데이터_추정매출_상주인구_소득소비_길단위인구_점포_영역.csv'
df = pd.read_csv(file_path, encoding='utf-8')
df['기준_년_코드'] = df['기준_년분기_코드'] // 10

print(f"✅ 데이터 로딩 완료: {len(df):,}건")

# =============================================================================
# 1. 상권 유형별 기술통계
# =============================================================================
print("\n" + "=" * 80)
print("1️⃣ 상권 유형별 매출 기술통계")
print("=" * 80)

area_types = ['골목상권', '발달상권', '전통시장', '관광특구']

for area in area_types:
    subset = df[df['상권_구분_코드_명'] == area]['당월_매출_금액']
    print(f"\n[{area}] (n={len(subset):,})")
    print(f"  평균: {subset.mean()/1e8:.2f}억원")
    print(f"  중앙값: {subset.median()/1e8:.2f}억원")
    print(f"  표준편차: {subset.std()/1e8:.2f}억원")
    print(f"  최소: {subset.min()/1e8:.4f}억원")
    print(f"  최대: {subset.max()/1e8:.2f}억원")
    print(f"  왜도: {subset.skew():.2f}")
    print(f"  로그 변환 후 왜도: {np.log1p(subset).skew():.2f}")

# =============================================================================
# 2. 상권 유형별 분위수 비교
# =============================================================================
print("\n" + "=" * 80)
print("2️⃣ 상권 유형별 분위수 비교 (억원)")
print("=" * 80)

quantiles = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
print(f"\n{'분위수':<10}", end="")
for area in area_types:
    print(f"{area:>12}", end="")
print()
print("-" * 60)

for q in quantiles:
    print(f"{int(q*100):>3}%      ", end="")
    for area in area_types:
        subset = df[df['상권_구분_코드_명'] == area]['당월_매출_금액']
        val = subset.quantile(q) / 1e8
        print(f"{val:>12.1f}", end="")
    print()

# =============================================================================
# 3. 시각화: 상권 유형별 매출 분포
# =============================================================================
print("\n" + "=" * 80)
print("3️⃣ 시각화 생성 중...")
print("=" * 80)

# 시각화 1: 상권 유형별 원본 매출 분포 (박스플롯)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

colors = {'골목상권': '#3498db', '발달상권': '#e74c3c', '전통시장': '#2ecc71', '관광특구': '#f39c12'}

for idx, area in enumerate(area_types):
    row, col = idx // 2, idx % 2
    ax = axes[row, col]
    subset = df[df['상권_구분_코드_명'] == area]['당월_매출_금액'] / 1e8
    
    # 히스토그램
    ax.hist(subset.clip(0, subset.quantile(0.95)), bins=50, color=colors[area], alpha=0.7, edgecolor='white')
    ax.axvline(subset.mean(), color='red', linestyle='--', linewidth=2, label=f'평균: {subset.mean():.1f}억')
    ax.axvline(subset.median(), color='green', linestyle='--', linewidth=2, label=f'중앙값: {subset.median():.1f}억')
    
    ax.set_title(f'{area} 매출 분포\n(n={len(subset):,}, 왜도={df[df["상권_구분_코드_명"]==area]["당월_매출_금액"].skew():.1f})', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('당월 매출 (억원)')
    ax.set_ylabel('빈도')
    ax.legend(fontsize=9)

plt.suptitle('상권 유형별 매출 분포 (상위 5% 제외)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig11_sales_dist_by_area_type.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 저장: fig11_sales_dist_by_area_type.png")

# 시각화 2: 상권 유형별 로그 변환 후 분포
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, area in enumerate(area_types):
    row, col = idx // 2, idx % 2
    ax = axes[row, col]
    subset = df[df['상권_구분_코드_명'] == area]['당월_매출_금액']
    log_subset = np.log1p(subset)
    
    ax.hist(log_subset, bins=50, color=colors[area], alpha=0.7, edgecolor='white')
    ax.axvline(log_subset.mean(), color='red', linestyle='--', linewidth=2, label=f'평균: {log_subset.mean():.1f}')
    ax.axvline(log_subset.median(), color='green', linestyle='--', linewidth=2, label=f'중앙값: {log_subset.median():.1f}')
    
    ax.set_title(f'{area} 로그 매출 분포\n(로그 왜도={log_subset.skew():.2f})', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('log(당월 매출)')
    ax.set_ylabel('빈도')
    ax.legend(fontsize=9)

plt.suptitle('상권 유형별 로그 변환 후 매출 분포', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig12_log_sales_dist_by_area_type.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 저장: fig12_log_sales_dist_by_area_type.png")

# 시각화 3: 박스플롯 비교
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 원본 박스플롯
box_data = [df[df['상권_구분_코드_명'] == area]['당월_매출_금액'] / 1e8 for area in area_types]
bp1 = axes[0].boxplot(box_data, labels=area_types, patch_artist=True, showfliers=False)
for patch, color in zip(bp1['boxes'], colors.values()):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_title('상권 유형별 매출 박스플롯 (이상치 제외)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('당월 매출 (억원)')
axes[0].tick_params(axis='x', rotation=15)

# 로그 박스플롯
log_box_data = [np.log1p(df[df['상권_구분_코드_명'] == area]['당월_매출_금액']) for area in area_types]
bp2 = axes[1].boxplot(log_box_data, labels=area_types, patch_artist=True, showfliers=False)
for patch, color in zip(bp2['boxes'], colors.values()):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_title('상권 유형별 로그 매출 박스플롯 (이상치 제외)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('log(당월 매출)')
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('fig13_boxplot_by_area_type.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 저장: fig13_boxplot_by_area_type.png")

# =============================================================================
# 4. 연도별 × 상권유형별 매출 추세
# =============================================================================
print("\n" + "=" * 80)
print("4️⃣ 연도별 × 상권유형별 매출 추세")
print("=" * 80)

yearly_area_sales = df.groupby(['기준_년_코드', '상권_구분_코드_명'])['당월_매출_금액'].mean().unstack() / 1e8
print("\n[연도별 상권유형별 평균 매출 (억원)]")
print(yearly_area_sales.round(1))

# 성장률 계산
print("\n[2021년 대비 2024년 성장률]")
for area in area_types:
    growth = (yearly_area_sales.loc[2024, area] - yearly_area_sales.loc[2021, area]) / yearly_area_sales.loc[2021, area] * 100
    print(f"  {area}: {growth:+.1f}%")

# 시각화 4: 연도별 추세 라인차트
fig, ax = plt.subplots(figsize=(12, 6))

for area in area_types:
    ax.plot(yearly_area_sales.index, yearly_area_sales[area], marker='o', linewidth=2, markersize=8, 
            label=area, color=colors[area])

ax.set_title('연도별 × 상권유형별 평균 매출 추세', fontsize=14, fontweight='bold')
ax.set_xlabel('연도')
ax.set_ylabel('평균 매출 (억원)')
ax.set_xticks([2021, 2022, 2023, 2024])
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig14_yearly_trend_by_area_type.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ 저장: fig14_yearly_trend_by_area_type.png")

# =============================================================================
# 5. 상권 유형별 로그 변환 효과 요약
# =============================================================================
print("\n" + "=" * 80)
print("5️⃣ 상권 유형별 로그 변환 효과 요약")
print("=" * 80)

print(f"\n{'상권유형':<12} {'원본왜도':>10} {'로그왜도':>10} {'개선율':>10}")
print("-" * 45)
for area in area_types:
    subset = df[df['상권_구분_코드_명'] == area]['당월_매출_금액']
    orig_skew = subset.skew()
    log_skew = np.log1p(subset).skew()
    improvement = (abs(orig_skew) - abs(log_skew)) / abs(orig_skew) * 100
    print(f"{area:<12} {orig_skew:>10.2f} {log_skew:>10.2f} {improvement:>9.1f}%")

print("\n" + "=" * 80)
print("✅ 상권 유형별 분석 완료!")
print("=" * 80)

