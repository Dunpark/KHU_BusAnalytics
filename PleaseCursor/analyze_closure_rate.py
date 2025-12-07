# -*- coding: utf-8 -*-
"""
analyze_closure_rate.py
업종별 폐업률 분석
- 2024년 4분기 데이터 기준
- 가중 평균 폐업률 계산
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("업종별 폐업률 분석")
print("=" * 60)

# ============================================================
# 1. 데이터 로드
# ============================================================
print("\n" + "=" * 60)
print("1. 데이터 로드")
print("=" * 60)

# 대용량 파일 로드 (필요한 컬럼만)
data_path = '../KHU_BusAnalytics/12_6/4개년_통합데이터_추정매출_상주인구_소득소비_길단위인구_점포_영역.csv'

# 필요한 컬럼
use_cols = ['기준_년분기_코드', '서비스_업종_코드_명', '점포_수', '폐업_점포_수', '개업_점포_수']

print("데이터 로딩 중...")
try:
    df = pd.read_csv(data_path, usecols=use_cols, encoding='utf-8')
except:
    try:
        df = pd.read_csv(data_path, usecols=use_cols, encoding='cp949')
    except:
        df = pd.read_csv(data_path, usecols=use_cols, encoding='utf-8-sig')

print(f"데이터 로드 완료: {df.shape}")

# ============================================================
# 2. 2024년 4분기 필터링
# ============================================================
print("\n" + "=" * 60)
print("2. 2024년 4분기 필터링")
print("=" * 60)

# 20244 분기 필터링
filtered_df = df[df['기준_년분기_코드'] == 20244].copy()
print(f"2024년 4분기 데이터: {len(filtered_df)} 행")

# 기본 통계
print(f"\n업종 수: {filtered_df['서비스_업종_코드_명'].nunique()}")
print(f"총 점포 수: {filtered_df['점포_수'].sum():,}")
print(f"총 폐업 점포 수: {filtered_df['폐업_점포_수'].sum():,}")

# ============================================================
# 3. 폐업률 계산
# ============================================================
print("\n" + "=" * 60)
print("3. 폐업률 계산")
print("=" * 60)

"""
폐업률 정의 및 정당성:
========================

정의: 폐업률 = Σ(폐업_점포_수) / Σ(점포_수) × 100

이 정의를 선택한 이유:

1. 가중 평균 효과:
   - 단순 평균이 아닌 총량 기반 계산
   - 상권 규모를 고려한 대표적인 폐업 리스크 파악
   - 점포수가 많은 상권의 폐업 현황이 더 큰 가중치

2. 업종별 비교 가능성:
   - 업종마다 상권 수와 점포 규모가 다름
   - 총량 기반 계산으로 업종 간 공정한 비교 가능

3. 시장 전체 관점:
   - 개별 상권이 아닌 업종 전체의 건전성 평가
   - 투자자/창업자 관점에서 업종별 리스크 비교에 유용

대안적 정의와 비교:
- 단순 평균 (상권별 폐업률의 산술평균): 소규모 상권이 과대평가될 수 있음
- 중앙값: 극단치에 강건하나 전체적인 규모 반영 어려움
"""

# 업종별 가중 평균 폐업률 계산
industry_closure = filtered_df.groupby('서비스_업종_코드_명').agg({
    '점포_수': 'sum',
    '폐업_점포_수': 'sum',
    '개업_점포_수': 'sum'
}).reset_index()

# 폐업률 계산
industry_closure['폐업률'] = (
    industry_closure['폐업_점포_수'] / industry_closure['점포_수'] * 100
).round(2)

# 개업률 계산
industry_closure['개업률'] = (
    industry_closure['개업_점포_수'] / industry_closure['점포_수'] * 100
).round(2)

# 순증감률 계산
industry_closure['순증감률'] = (industry_closure['개업률'] - industry_closure['폐업률']).round(2)

# 상권 수 계산
industry_count = filtered_df.groupby('서비스_업종_코드_명').size().reset_index(name='상권_수')
industry_closure = industry_closure.merge(industry_count, on='서비스_업종_코드_명')

# 폐업률 순으로 정렬
industry_closure_sorted = industry_closure.sort_values('폐업률', ascending=False)

print("\n업종별 폐업률 (상위 20):")
print(industry_closure_sorted[['서비스_업종_코드_명', '점포_수', '폐업_점포_수', '폐업률', '상권_수']].head(20).to_string(index=False))

# ============================================================
# 4. 폐업률 상위 10개 업종 분석
# ============================================================
print("\n" + "=" * 60)
print("4. 폐업률 상위 10개 업종")
print("=" * 60)

top10_closure = industry_closure_sorted.head(10)

print("\n폐업률 상위 10개 업종:")
print("-" * 80)
for i, row in top10_closure.iterrows():
    print(f"{row['서비스_업종_코드_명']}")
    print(f"  - 폐업률: {row['폐업률']:.2f}%")
    print(f"  - 총 점포수: {row['점포_수']:,}")
    print(f"  - 폐업 점포수: {row['폐업_점포_수']:,}")
    print(f"  - 상권 수: {row['상권_수']}")
    print()

# ============================================================
# 5. 폐업률 vs 개업률 비교
# ============================================================
print("\n" + "=" * 60)
print("5. 폐업률 vs 개업률 비교")
print("=" * 60)

# 순증감률 기준 정렬
industry_net_sorted = industry_closure.sort_values('순증감률')

print("\n순증감률 하위 10개 업종 (폐업 > 개업):")
print(industry_net_sorted[['서비스_업종_코드_명', '개업률', '폐업률', '순증감률']].head(10).to_string(index=False))

print("\n순증감률 상위 10개 업종 (개업 > 폐업):")
print(industry_net_sorted[['서비스_업종_코드_명', '개업률', '폐업률', '순증감률']].tail(10).to_string(index=False))

# ============================================================
# 6. 카페/음료 업종 분석
# ============================================================
print("\n" + "=" * 60)
print("6. 카페/음료 업종 분석")
print("=" * 60)

# 커피-음료 업종 찾기
coffee_industry = industry_closure[industry_closure['서비스_업종_코드_명'].str.contains('커피|음료|카페', na=False)]

if len(coffee_industry) > 0:
    print("\n커피/음료 관련 업종:")
    print(coffee_industry[['서비스_업종_코드_명', '점포_수', '폐업률', '개업률', '순증감률']].to_string(index=False))
    
    # 전체 대비 순위
    total_industries = len(industry_closure)
    for _, row in coffee_industry.iterrows():
        rank = (industry_closure_sorted['서비스_업종_코드_명'] == row['서비스_업종_코드_명']).idxmax()
        actual_rank = list(industry_closure_sorted['서비스_업종_코드_명']).index(row['서비스_업종_코드_명']) + 1
        print(f"\n{row['서비스_업종_코드_명']} 폐업률 순위: {actual_rank}/{total_industries}")
else:
    print("커피/음료 관련 업종을 찾을 수 없습니다.")

# ============================================================
# 7. 시각화
# ============================================================
print("\n" + "=" * 60)
print("7. 시각화")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 7-1. 폐업률 상위 15개 업종
ax1 = axes[0, 0]
top15 = industry_closure_sorted.head(15)
colors = plt.cm.Reds(np.linspace(0.3, 0.9, 15))
bars = ax1.barh(top15['서비스_업종_코드_명'], top15['폐업률'], color=colors)
ax1.set_xlabel('폐업률 (%)')
ax1.set_title('폐업률 상위 15개 업종', fontsize=14, fontweight='bold')
ax1.invert_yaxis()
for bar, val in zip(bars, top15['폐업률']):
    ax1.text(val + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
             va='center', fontsize=9)

# 7-2. 순증감률 분포
ax2 = axes[0, 1]
ax2.hist(industry_closure['순증감률'], bins=30, color='#2E86AB', alpha=0.7, edgecolor='white')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='균형점')
ax2.axvline(x=industry_closure['순증감률'].mean(), color='green', linestyle='--', 
            linewidth=2, label=f'평균: {industry_closure["순증감률"].mean():.2f}%')
ax2.set_xlabel('순증감률 (개업률 - 폐업률)')
ax2.set_ylabel('업종 수')
ax2.set_title('업종별 순증감률 분포', fontsize=14, fontweight='bold')
ax2.legend()

# 7-3. 개업률 vs 폐업률 산점도
ax3 = axes[1, 0]
scatter = ax3.scatter(industry_closure['개업률'], industry_closure['폐업률'], 
                      s=industry_closure['점포_수']/1000, alpha=0.6, c='#2E86AB')
ax3.plot([0, max(industry_closure['개업률'])], [0, max(industry_closure['개업률'])], 
         'r--', label='개업률 = 폐업률')
ax3.set_xlabel('개업률 (%)')
ax3.set_ylabel('폐업률 (%)')
ax3.set_title('개업률 vs 폐업률\n(원 크기 = 점포 수)', fontsize=14, fontweight='bold')
ax3.legend()

# 7-4. 점포수 vs 폐업률
ax4 = axes[1, 1]
ax4.scatter(industry_closure['점포_수'], industry_closure['폐업률'], 
            alpha=0.6, c='#E94F37')
ax4.set_xlabel('총 점포 수')
ax4.set_ylabel('폐업률 (%)')
ax4.set_title('점포 규모 vs 폐업률', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('closure_rate_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[저장] closure_rate_analysis.png")

# ============================================================
# 8. 결과 저장
# ============================================================
print("\n" + "=" * 60)
print("8. 결과 저장")
print("=" * 60)

# 전체 업종별 폐업률 저장
industry_closure_sorted.to_csv('closure_rate_by_industry.csv', index=False, encoding='utf-8-sig')
print("\n[저장] closure_rate_by_industry.csv")

# ============================================================
# 9. 주요 발견 사항
# ============================================================
print("\n" + "=" * 60)
print("9. 주요 발견 사항")
print("=" * 60)

print(f"\n[전체 현황]")
print(f"- 분석 업종 수: {len(industry_closure)}")
print(f"- 전체 평균 폐업률: {industry_closure['폐업률'].mean():.2f}%")
print(f"- 전체 평균 개업률: {industry_closure['개업률'].mean():.2f}%")
print(f"- 전체 평균 순증감률: {industry_closure['순증감률'].mean():.2f}%")

print(f"\n[폐업률 상위 3개 업종]")
for i, row in top10_closure.head(3).iterrows():
    print(f"- {row['서비스_업종_코드_명']}: {row['폐업률']:.2f}%")

print(f"\n[폐업률 하위 3개 업종]")
for i, row in industry_closure_sorted.tail(3).iterrows():
    print(f"- {row['서비스_업종_코드_명']}: {row['폐업률']:.2f}%")

print("\n" + "=" * 60)
print("폐업률 분석 완료!")
print("=" * 60)

