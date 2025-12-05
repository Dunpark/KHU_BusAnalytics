# -*- coding: utf-8 -*-
import pandas as pd

print("=" * 60)
print("원본 대비 최종 통합 데이터 비교")
print("=" * 60)

# 원본 파일 (4개년추정매출.csv)
df_original = pd.read_csv('Merged_datasets/4개년추정매출.csv', encoding='utf-8-sig', low_memory=False)
original_rows = len(df_original)
print(f"\n[원본] 4개년추정매출.csv: {original_rows:,}행")

# 최종 통합 파일
df_final = pd.read_csv('Merged_datasets/4개년_통합데이터_추정매출_상주인구_소득소비_길단위인구_점포_영역.csv', 
                       encoding='utf-8-sig', low_memory=False)
final_rows = len(df_final)
print(f"[최종] 통합데이터_영역: {final_rows:,}행")

# 비교
lost_rows = original_rows - final_rows
lost_pct = lost_rows / original_rows * 100
kept_pct = final_rows / original_rows * 100

print(f"\n{'='*50}")
print(f"누락된 행: {lost_rows:,}행")
print(f"결측비율: {lost_pct:.2f}%")
print(f"유지율: {kept_pct:.2f}%")
print(f"{'='*50}")

print(f"\n[최종 데이터 정보]")
print(f"  총 컬럼 수: {len(df_final.columns)}")
print(f"  결측치 수: {df_final.isna().sum().sum():,}")

