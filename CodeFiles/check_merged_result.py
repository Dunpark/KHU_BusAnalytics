# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv('Merged_datasets/4개년추정매출.csv', encoding='utf-8-sig')

print("=" * 60)
print("4개년 추정매출 병합 결과")
print("=" * 60)
print(f"\n총 행 수: {len(df):,}")
print(f"총 컬럼 수: {len(df.columns)}")
print(f"결측치: {df.isna().sum().sum():,}")

# 연도별 분포
print("\n[연도별 분포]")
df['연도'] = df['기준_년분기_코드'].astype(str).str[:4]
for y, cnt in df.groupby('연도').size().sort_index().items():
    print(f"  {y}년: {cnt:,}행")

# 분기별 분포
print("\n[분기별 분포]")
for q, cnt in df.groupby('기준_년분기_코드').size().sort_index().items():
    print(f"  {q}: {cnt:,}행")

# 공통 조합 수
unique_combos = df[['상권_코드', '서비스_업종_코드']].drop_duplicates()
print(f"\n공통 (상권, 업종) 조합: {len(unique_combos):,}개")

