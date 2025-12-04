# -*- coding: utf-8 -*-
import pandas as pd

print("=" * 70)
print("결측치 원인 분석")
print("=" * 70)

# 1. 소득소비-상권 원본 결측치 확인
print("\n[1] 소득소비-상권 원본 결측치")
print("-" * 50)
try:
    income_df = pd.read_csv('Data_Raw_new/서울시 상권분석서비스(소득소비-상권).csv', encoding='utf-8-sig', low_memory=False)
except:
    income_df = pd.read_csv('Data_Raw_new/서울시 상권분석서비스(소득소비-상권).csv', encoding='cp949', low_memory=False)

print(f"총 행 수: {len(income_df):,}")
income_missing = income_df.isna().sum()
income_missing_cols = income_missing[income_missing > 0]
print(f"\n결측이 있는 컬럼:")
for col, cnt in income_missing_cols.items():
    print(f"  {col}: {cnt:,}개 ({cnt/len(income_df)*100:.1f}%)")

# 2. 상주인구-상권 원본 결측치 확인
print("\n[2] 상주인구-상권 원본 결측치")
print("-" * 50)
try:
    pop_df = pd.read_csv('Data_Raw_new/서울시 상권분석서비스(상주인구-상권).csv', encoding='utf-8-sig', low_memory=False)
except:
    pop_df = pd.read_csv('Data_Raw_new/서울시 상권분석서비스(상주인구-상권).csv', encoding='cp949', low_memory=False)

print(f"총 행 수: {len(pop_df):,}")
pop_missing = pop_df.isna().sum()
pop_missing_cols = pop_missing[pop_missing > 0]
if len(pop_missing_cols) > 0:
    print(f"\n결측이 있는 컬럼:")
    for col, cnt in pop_missing_cols.items():
        print(f"  {col}: {cnt:,}개 ({cnt/len(pop_df)*100:.1f}%)")
else:
    print("결측 없음")

# 3. Merge 후 결측치 확인
print("\n[3] Merge 후 결측치 (4개년추정매출_소득소비_상주인구.csv)")
print("-" * 50)
merged_df = pd.read_csv('Merged_datasets/4개년추정매출_소득소비_상주인구.csv', encoding='utf-8-sig', low_memory=False)
print(f"총 행 수: {len(merged_df):,}")

merged_missing = merged_df.isna().sum()
merged_missing_cols = merged_missing[merged_missing > 0].sort_values(ascending=False)
print(f"\n결측이 있는 컬럼 ({len(merged_missing_cols)}개):")
for col, cnt in merged_missing_cols.items():
    print(f"  {col}: {cnt:,}개 ({cnt/len(merged_df)*100:.1f}%)")

# 4. 결측치 발생 원인 분석
print("\n" + "=" * 70)
print("결측치 발생 원인 요약")
print("=" * 70)

print("""
[원인 1] 원본 데이터의 결측치
-----------------------------------------
소득소비-상권 원본에 이미 결측치가 존재합니다:
  - 월_평균_소득_금액: 174개 (0.4%)
  - 지출 관련 컬럼들: 각 904개 (2.1%)

이 결측치는 해당 상권의 소득/소비 데이터가 
수집되지 않았거나 집계가 불가능한 경우입니다.

[원인 2] Merge 시 결측치 증폭
-----------------------------------------
소득소비-상권의 1개 행이 4개년추정매출의 여러 행과 매칭됩니다.
(같은 상권에 여러 업종이 존재하므로)

예시:
  소득소비-상권: 상권A의 소득=결측 (1개)
       ↓ Merge
  4개년추정매출: 상권A × 10개 업종 = 10개 행에 결측 전파
""")

# 결측치 증폭 비율 계산
income_original_missing = 174 + 904  # 대략적인 원본 결측
merged_total_missing = merged_missing.sum()
print(f"\n[수치 확인]")
print(f"  소득소비-상권 원본 결측치 (행 기준): ~{income_df.isna().any(axis=1).sum():,}행")
print(f"  Merge 후 총 결측치 (셀 기준): {merged_total_missing:,}개")

