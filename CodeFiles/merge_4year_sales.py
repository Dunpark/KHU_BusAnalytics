# -*- coding: utf-8 -*-
import pandas as pd
import os

# 파일 경로
base_path = 'Data_Raw_new'
files = {
    '2021': '서울시_상권분석서비스(추정매출-상권)_2021년.csv',
    '2022': '서울시_상권분석서비스(추정매출-상권)_2022년.csv',
    '2023': '서울시_상권분석서비스(추정매출-상권)_2023년.csv',
    '2024': '서울시 상권분석서비스(추정매출-상권)_2024년.csv'
}

# 조인 키 컬럼
KEY_COLS = ['상권_코드', '서비스_업종_코드']

print("=" * 70)
print("4개년 추정매출 데이터 Inner Join 병합 (모든 분기 공통)")
print("=" * 70)

# 1. 파일 읽기 및 합치기
print("\n[1/5] 파일 읽기...")
all_dfs = []
for year, filename in files.items():
    filepath = os.path.join(base_path, filename)
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='cp949', low_memory=False)
    all_dfs.append(df)
    print(f"  {year}년: {len(df):,}행")

# 전체 데이터 합치기
combined_df = pd.concat(all_dfs, ignore_index=True)
print(f"\n  → 전체 합산: {len(combined_df):,}행")

# 2. 전체 분기 확인
all_quarters = sorted(combined_df['기준_년분기_코드'].unique())
print(f"\n[2/5] 전체 분기 확인...")
print(f"  총 분기 수: {len(all_quarters)}개")
print(f"  분기 목록: {all_quarters}")

# 3. 각 분기별 고유 조합 추출 및 교집합 계산
print("\n[3/5] 분기별 고유 조합 추출 및 교집합 계산...")

# 첫 번째 분기의 조합으로 시작
first_quarter = all_quarters[0]
first_df = combined_df[combined_df['기준_년분기_코드'] == first_quarter]
common_keys = set(first_df[KEY_COLS].apply(tuple, axis=1))
print(f"  {first_quarter}: {len(common_keys):,}개 고유 조합")

# 나머지 분기와 교집합 계산
for quarter in all_quarters[1:]:
    quarter_df = combined_df[combined_df['기준_년분기_코드'] == quarter]
    quarter_keys = set(quarter_df[KEY_COLS].apply(tuple, axis=1))
    common_keys = common_keys.intersection(quarter_keys)
    print(f"  {quarter}: {len(quarter_keys):,}개 → 교집합: {len(common_keys):,}개")

print(f"\n  → 모든 {len(all_quarters)}개 분기 공통 조합: {len(common_keys):,}개")

# 4. 공통 조합만 필터링
print("\n[4/5] 데이터 필터링...")
combined_key = combined_df[KEY_COLS].apply(tuple, axis=1)
filtered_df = combined_df[combined_key.isin(common_keys)].copy()

print(f"  필터링 전: {len(combined_df):,}행")
print(f"  필터링 후: {len(filtered_df):,}행")
print(f"  제외된 행: {len(combined_df) - len(filtered_df):,}행 ({(len(combined_df) - len(filtered_df)) / len(combined_df) * 100:.1f}%)")

# 5. 결과 저장
print("\n[5/5] 결과 저장...")
output_dir = 'Merged_datasets'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, '4개년추정매출_all_quarters.csv')
filtered_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"  저장 완료: {output_path}")

# 최종 통계
print("\n" + "=" * 70)
print("최종 병합 결과 요약")
print("=" * 70)
print(f"  총 행 수: {len(filtered_df):,}")
print(f"  총 컬럼 수: {len(filtered_df.columns)}")
print(f"  모든 분기 공통 (상권, 업종) 조합: {len(common_keys):,}개")
print(f"  포함 분기: {len(all_quarters)}개 ({all_quarters[0]} ~ {all_quarters[-1]})")

# 연도별 분기 분포
print("\n  [연도별 행 분포]")
filtered_df['연도'] = filtered_df['기준_년분기_코드'].astype(str).str[:4]
year_dist = filtered_df.groupby('연도').size()
for y, cnt in year_dist.items():
    print(f"    {y}년: {cnt:,}행")

# 분기별 분포
print("\n  [분기별 행 분포]")
quarter_dist = filtered_df.groupby('기준_년분기_코드').size()
for q, cnt in quarter_dist.items():
    print(f"    {q}: {cnt:,}행")

# 결측치 확인
missing = filtered_df.isna().sum().sum()
print(f"\n  전체 결측치: {missing:,}개")

# 검증: 각 분기별 고유 조합 수 확인
print("\n  [검증] 각 분기별 고유 조합 수:")
for q in all_quarters:
    q_df = filtered_df[filtered_df['기준_년분기_코드'] == q]
    q_combos = len(q_df[KEY_COLS].drop_duplicates())
    print(f"    {q}: {q_combos:,}개")
