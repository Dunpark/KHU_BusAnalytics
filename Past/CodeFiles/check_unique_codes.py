# -*- coding: utf-8 -*-
import pandas as pd

try:
    df = pd.read_csv('Data_Raw_new/서울시 상권분석서비스(추정매출-상권)_2024년.csv', encoding='utf-8-sig')
except UnicodeDecodeError:
    df = pd.read_csv('Data_Raw_new/서울시 상권분석서비스(추정매출-상권)_2024년.csv', encoding='cp949')
print(f"고유 상권_코드 개수: {df['상권_코드'].nunique():,}개")
print(f"총 행 수: {len(df):,}행")

