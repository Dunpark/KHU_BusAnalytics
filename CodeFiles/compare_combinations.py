# -*- coding: utf-8 -*-
import pandas as pd

print("=" * 70)
print("조합 비교 분석")
print("=" * 70)

# 1. 4개년추정매출 데이터 로드
sales_df = pd.read_csv('Merged_datasets/4개년추정매출_all_quarters.csv', encoding='utf-8-sig', low_memory=False)
print(f"\n[4개년추정매출_all_quarters.csv]")
print(f"  행 수: {len(sales_df):,}")

# 2. 소득소비-상권 데이터 로드
try:
    income_df = pd.read_csv('Data_Raw_new/서울시 상권분석서비스(소득소비-상권).csv', encoding='utf-8-sig', low_memory=False)
except UnicodeDecodeError:
    income_df = pd.read_csv('Data_Raw_new/서울시 상권분석서비스(소득소비-상권).csv', encoding='cp949', low_memory=False)
print(f"\n[소득소비-상권.csv]")
print(f"  행 수: {len(income_df):,}")

# ========================================
# 분석 1: (기준_년분기_코드, 상권_구분_코드, 상권_코드) 조합 비교
# ※ 서비스_업종_코드는 소득소비-상권에 없으므로 제외
# ========================================
print("\n" + "=" * 70)
print("분석: (기준_년분기_코드, 상권_구분_코드, 상권_코드) 조합 비교")
print("=" * 70)

KEY_COLS = ['기준_년분기_코드', '상권_구분_코드', '상권_코드']

# 4개년추정매출에서 고유 조합 추출
sales_combos = set(sales_df[KEY_COLS].drop_duplicates().apply(tuple, axis=1))
print(f"\n[4개년추정매출] 고유 조합 수: {len(sales_combos):,}개")

# 소득소비-상권에서 고유 조합 추출
income_combos = set(income_df[KEY_COLS].drop_duplicates().apply(tuple, axis=1))
print(f"[소득소비-상권] 고유 조합 수: {len(income_combos):,}개")

# 교집합
common = sales_combos.intersection(income_combos)
print(f"\n[공통 조합]: {len(common):,}개")

# 4개년추정매출에만 있는 조합
only_in_sales = sales_combos - income_combos
print(f"[4개년추정매출에만 있는 조합]: {len(only_in_sales):,}개 ({len(only_in_sales)/len(sales_combos)*100:.1f}%)")

# 소득소비-상권에만 있는 조합
only_in_income = income_combos - sales_combos
print(f"[소득소비-상권에만 있는 조합]: {len(only_in_income):,}개")

# ========================================
# 누락된 조합 상세 분석
# ========================================
if only_in_sales:
    print("\n" + "-" * 70)
    print("4개년추정매출에는 있지만 소득소비-상권에 없는 조합 분석")
    print("-" * 70)
    
    missing_df = pd.DataFrame(list(only_in_sales), columns=KEY_COLS)
    
    # 분기별 누락 수
    print("\n[분기별 누락 조합 수]")
    quarter_missing = missing_df.groupby('기준_년분기_코드').size().sort_index()
    for q, cnt in quarter_missing.items():
        print(f"  {q}: {cnt:,}개")
    
    # 상권구분별 누락 수
    print("\n[상권구분별 누락 조합 수]")
    type_missing = missing_df.groupby('상권_구분_코드').size()
    for t, cnt in type_missing.items():
        print(f"  {t}: {cnt:,}개")

# ========================================
# 참고: 분기 범위 비교
# ========================================
print("\n" + "=" * 70)
print("분기 범위 비교")
print("=" * 70)

sales_quarters = sorted(sales_df['기준_년분기_코드'].unique())
income_quarters = sorted(income_df['기준_년분기_코드'].unique())

print(f"\n[4개년추정매출 분기]: {len(sales_quarters)}개")
print(f"  {sales_quarters}")

print(f"\n[소득소비-상권 분기]: {len(income_quarters)}개")
print(f"  {income_quarters}")

# 공통 분기
common_quarters = set(sales_quarters).intersection(set(income_quarters))
print(f"\n[공통 분기]: {len(common_quarters)}개")
print(f"  {sorted(common_quarters)}")

