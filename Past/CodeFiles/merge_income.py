# -*- coding: utf-8 -*-
import pandas as pd

print("=" * 70)
print("4개년추정매출 + 소득소비-상권 Merge")
print("=" * 70)

# 1. 데이터 로드
print("\n[1/4] 데이터 로드...")
sales_df = pd.read_csv('Merged_datasets/4개년추정매출_all_quarters.csv', encoding='utf-8-sig', low_memory=False)
print(f"  4개년추정매출: {len(sales_df):,}행, {len(sales_df.columns)}컬럼")

try:
    income_df = pd.read_csv('Data_Raw_new/서울시 상권분석서비스(소득소비-상권).csv', encoding='utf-8-sig', low_memory=False)
except UnicodeDecodeError:
    income_df = pd.read_csv('Data_Raw_new/서울시 상권분석서비스(소득소비-상권).csv', encoding='cp949', low_memory=False)
print(f"  소득소비-상권: {len(income_df):,}행, {len(income_df.columns)}컬럼")

# 2. Merge 키 설정
KEY_COLS = ['기준_년분기_코드', '상권_구분_코드', '상권_코드']

# 소득소비-상권에서 중복 컬럼 제거 (상권_구분_코드_명, 상권_코드_명은 sales_df에 이미 있음)
income_cols_to_add = [c for c in income_df.columns if c not in ['상권_구분_코드_명', '상권_코드_명']]
income_df_clean = income_df[income_cols_to_add]
print(f"\n[2/4] Merge 준비...")
print(f"  조인 키: {KEY_COLS}")
print(f"  추가될 컬럼: {len(income_df_clean.columns) - len(KEY_COLS)}개")

# 3. Inner Join 수행
print("\n[3/4] Inner Join 수행...")
merged_df = pd.merge(sales_df, income_df_clean, on=KEY_COLS, how='inner')

print(f"\n  [Merge 결과]")
print(f"  원본 행 수: {len(sales_df):,}행")
print(f"  Merge 후 행 수: {len(merged_df):,}행")
print(f"  누락된 행 수: {len(sales_df) - len(merged_df):,}행 ({(len(sales_df) - len(merged_df))/len(sales_df)*100:.2f}%)")
print(f"  컬럼 수: {len(sales_df.columns)} → {len(merged_df.columns)}")

# 누락된 행 분석
if len(sales_df) > len(merged_df):
    print("\n  [누락된 행 상세]")
    # 누락된 조합 찾기
    sales_keys = set(sales_df[KEY_COLS].apply(tuple, axis=1))
    merged_keys = set(merged_df[KEY_COLS].apply(tuple, axis=1))
    missing_keys = sales_keys - merged_keys
    
    if missing_keys:
        missing_df = pd.DataFrame(list(missing_keys), columns=KEY_COLS)
        
        # 분기별
        print("\n  분기별 누락:")
        for q, cnt in missing_df.groupby('기준_년분기_코드').size().sort_index().items():
            print(f"    {q}: {cnt}개 조합")
        
        # 상권구분별
        print("\n  상권구분별 누락:")
        for t, cnt in missing_df.groupby('상권_구분_코드').size().items():
            print(f"    {t}: {cnt}개 조합")

# 4. 결과 저장
print("\n[4/4] 결과 저장...")
output_path = 'Merged_datasets/4개년추정매출_소득소비.csv'
merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"  저장 완료: {output_path}")

# 최종 요약
print("\n" + "=" * 70)
print("최종 요약")
print("=" * 70)
print(f"  총 행 수: {len(merged_df):,}")
print(f"  총 컬럼 수: {len(merged_df.columns)}")
print(f"  결측치: {merged_df.isna().sum().sum():,}개")

# 추가된 소득소비 컬럼 목록
added_cols = [c for c in merged_df.columns if c not in sales_df.columns]
print(f"\n  [추가된 소득소비 컬럼] ({len(added_cols)}개)")
for col in added_cols:
    print(f"    - {col}")

