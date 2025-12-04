# -*- coding: utf-8 -*-
"""
서울시 상권분석서비스(점포-상권) 4개년 파일 단순 합치기 (행 추가 방식)
"""

import pandas as pd
import os

print("=" * 70)
print("점포-상권 4개년 데이터 합치기")
print("=" * 70)

# 파일 경로
base_path = 'Data_Raw_new'
files = {
    '2021': '서울시_상권분석서비스(점포-상권)_2021년.csv',
    '2022': '서울시_상권분석서비스(점포-상권)_2022년.csv',
    '2023': '서울시_상권분석서비스(점포-상권)_2023년.csv',
    '2024': '서울시 상권분석서비스(점포-상권)_2024년.csv'
}

# 각 파일 로드
print("\n[1/3] 파일 로드...")
dfs = []
total_rows = 0

for year, filename in files.items():
    filepath = os.path.join(base_path, filename)
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='cp949', low_memory=False)
    
    dfs.append(df)
    total_rows += len(df)
    print(f"  {year}년: {len(df):,}행, {len(df.columns)}컬럼")

# 행 합치기 (concat)
print("\n[2/3] 데이터 합치기...")
df_merged = pd.concat(dfs, ignore_index=True)

print(f"\n  개별 파일 합계: {total_rows:,}행")
print(f"  합친 결과: {df_merged.shape[0]:,}행, {df_merged.shape[1]}컬럼")

# 저장
print("\n[3/3] 저장...")
output_path = 'Merged_datasets/merged_점포.csv'
df_merged.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"  저장 완료: {output_path}")

# 결과 요약
print("\n" + "=" * 70)
print("결과 요약")
print("=" * 70)
print(f"  총 행 수: {df_merged.shape[0]:,}")
print(f"  총 컬럼 수: {df_merged.shape[1]}")

# 분기별 분포
print("\n[분기별 분포]")
quarter_dist = df_merged.groupby('기준_년분기_코드').size().sort_index()
for q, cnt in quarter_dist.items():
    print(f"  {q}: {cnt:,}행")

