# -*- coding: utf-8 -*-
import pandas as pd

print("=" * 70)
print("1. 소득소비-상권 데이터셋 정밀 분석")
print("=" * 70)

# 소득소비-상권 파일 읽기
try:
    income_df = pd.read_csv('Data_Raw_new/서울시 상권분석서비스(소득소비-상권).csv', encoding='utf-8-sig', low_memory=False)
except UnicodeDecodeError:
    income_df = pd.read_csv('Data_Raw_new/서울시 상권분석서비스(소득소비-상권).csv', encoding='cp949', low_memory=False)

print(f"\n[기본 정보]")
print(f"  행 수: {len(income_df):,}")
print(f"  컬럼 수: {len(income_df.columns)}")

print(f"\n[컬럼 목록]")
for i, col in enumerate(income_df.columns):
    print(f"  {i+1:2}. {col}")

print(f"\n[기준_년분기_코드 범위]")
quarters = sorted(income_df['기준_년분기_코드'].unique())
print(f"  총 분기 수: {len(quarters)}개")
print(f"  분기 목록: {quarters}")

print(f"\n[상권_구분_코드 분포]")
if '상권_구분_코드' in income_df.columns:
    print(income_df['상권_구분_코드'].value_counts())
elif '상권_구분_코드_명' in income_df.columns:
    print(income_df['상권_구분_코드_명'].value_counts())

# 서비스_업종_코드 확인
print(f"\n[서비스_업종_코드 존재 여부]")
if '서비스_업종_코드' in income_df.columns:
    print(f"  서비스_업종_코드 컬럼 있음")
    print(f"  고유 업종 수: {income_df['서비스_업종_코드'].nunique():,}개")
else:
    print(f"  서비스_업종_코드 컬럼 없음!")
    code_cols = [c for c in income_df.columns if '코드' in c]
    print(f"  코드 관련 컬럼: {code_cols}")

print(f"\n[결측치]")
missing = income_df.isna().sum()
missing_cols = missing[missing > 0]
if len(missing_cols) > 0:
    print("  결측이 있는 컬럼:")
    for col, cnt in missing_cols.items():
        print(f"    {col}: {cnt:,} ({cnt/len(income_df)*100:.1f}%)")
else:
    print("  결측 없음")

print(f"\n[샘플 데이터 (첫 5행)]")
print(income_df.head())

