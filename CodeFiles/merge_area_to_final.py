# -*- coding: utf-8 -*-
"""
기존 통합 데이터에 영역-상권 데이터를 추가로 병합하는 코드
- 영역-상권은 기준_년분기_코드가 없으므로 상권_코드만으로 조인
"""

import pandas as pd

# =============================================================================
# 1. 데이터 로드
# =============================================================================
print("=" * 70)
print("[1/5] 데이터 로드")
print("=" * 70)

# 기존 통합 데이터
df_merged = pd.read_csv(
    'Merged_datasets/4개년_통합데이터_추정매출_상주인구_소득소비_길단위인구_점포.csv',
    encoding='utf-8-sig',
    low_memory=False
)
print(f"기존 통합 데이터 로드 완료: {df_merged.shape}")

# 영역-상권 데이터
try:
    df_area = pd.read_csv(
        'Data_Raw_new/서울시 상권분석서비스(영역-상권).csv',
        encoding='utf-8-sig',
        low_memory=False
    )
except UnicodeDecodeError:
    df_area = pd.read_csv(
        'Data_Raw_new/서울시 상권분석서비스(영역-상권).csv',
        encoding='cp949',
        low_memory=False
    )
print(f"영역-상권 데이터 로드 완료: {df_area.shape}")

# 원본 데이터 행 수 (비교용)
ORIGINAL_ROWS = 290417  # 4개년추정매출_all_quarters.csv

# 영역-상권 컬럼 확인
print(f"\n영역-상권 컬럼: {list(df_area.columns)}")

# =============================================================================
# 2. 조인 키 설정
# =============================================================================
print("\n" + "=" * 70)
print("[2/5] 조인 키 설정")
print("=" * 70)

# 영역-상권은 기준_년분기_코드가 없으므로 상권_코드만으로 조인
join_keys = ["상권_코드"]
print(f"조인 키: {join_keys} (영역-상권은 기준_년분기_코드 없음)")

# =============================================================================
# 3. 영역-상권 데이터 Merge (LEFT JOIN)
# =============================================================================
print("\n" + "=" * 70)
print("[3/5] 영역-상권 데이터 Merge")
print("=" * 70)

# 중복 컬럼 제거
area_cols_to_drop = [col for col in df_area.columns 
                     if col not in join_keys and col in df_merged.columns]
print(f"제거할 중복 컬럼: {area_cols_to_drop}")

df_area_clean = df_area.drop(columns=area_cols_to_drop, errors='ignore')

# 영역 관련 컬럼 리스트
area_data_cols = [col for col in df_area_clean.columns if col not in join_keys]
print(f"영역 관련 컬럼 수: {len(area_data_cols)}개")
print(f"영역 관련 컬럼: {area_data_cols}")

# LEFT JOIN 수행
df_tmp = pd.merge(
    df_merged,
    df_area_clean,
    on=join_keys,
    how='left',
    suffixes=('', '_area')
)

print(f"\nMerge 결과:")
print(f"  df_merged: {df_merged.shape[0]:,}행")
print(f"  df_tmp (merge 후): {df_tmp.shape[0]:,}행")
print(f"  컬럼 수 변화: {df_merged.shape[1]} → {df_tmp.shape[1]}")

# =============================================================================
# 4. 결측치 기반 필터링 (상권_코드, 서비스_업종_코드 그룹별)
# =============================================================================
print("\n" + "=" * 70)
print("[4/5] 결측치 기반 필터링")
print("=" * 70)

GROUP_KEYS = ["상권_코드", "서비스_업종_코드"]

# 결측치 확인 대상: 영역 관련 컬럼
check_cols = [col for col in df_tmp.columns if col in area_data_cols]
print(f"결측치 확인 대상 컬럼 수: {len(check_cols)}개")

# 각 행에서 check_cols 중 하나라도 NaN이면 표시
df_tmp['has_missing'] = df_tmp[check_cols].isna().any(axis=1)

# 그룹별로 결측치 있는 행이 하나라도 있으면 그룹 전체 제거
df_tmp['group_has_missing'] = df_tmp.groupby(GROUP_KEYS)['has_missing'].transform('sum')

df_final = df_tmp[df_tmp['group_has_missing'] == 0].copy()
df_final = df_final.drop(columns=['has_missing', 'group_has_missing'])

# 결과 출력
removed_rows = df_tmp.shape[0] - df_final.shape[0]
removed_groups = df_tmp[df_tmp['group_has_missing'] > 0][GROUP_KEYS].drop_duplicates().shape[0]

print(f"\n필터링 결과:")
print(f"  df_tmp (필터링 전): {df_tmp.shape[0]:,}행")
print(f"  df_final (필터링 후): {df_final.shape[0]:,}행")
print(f"  이번 단계 제거된 행: {removed_rows:,}행 ({removed_rows/df_tmp.shape[0]*100:.2f}%)")
print(f"  이번 단계 제거된 그룹: {removed_groups:,}개")

# =============================================================================
# 5. 결과 저장 및 최종 비교
# =============================================================================
print("\n" + "=" * 70)
print("[5/5] 결과 저장 및 원본 대비 비교")
print("=" * 70)

output_path = 'Merged_datasets/4개년_통합데이터_추정매출_상주인구_소득소비_길단위인구_점포_영역.csv'
df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n저장 완료: {output_path}")

# 원본 대비 비교
print("\n" + "=" * 70)
print("★ 원본 대비 최종 비교 ★")
print("=" * 70)

total_lost = ORIGINAL_ROWS - df_final.shape[0]
print(f"\n[원본] 4개년추정매출_all_quarters.csv: {ORIGINAL_ROWS:,}행")
print(f"[최종] 통합데이터_영역 추가: {df_final.shape[0]:,}행")
print(f"\n{'='*40}")
print(f"총 누락된 행: {total_lost:,}행")
print(f"결측비율: {total_lost/ORIGINAL_ROWS*100:.2f}%")
print(f"유지율: {df_final.shape[0]/ORIGINAL_ROWS*100:.2f}%")
print(f"{'='*40}")

# 최종 데이터 정보
print(f"\n[최종 데이터 정보]")
print(f"  총 행 수: {df_final.shape[0]:,}")
print(f"  총 컬럼 수: {df_final.shape[1]}")
print(f"  결측치 수: {df_final.isna().sum().sum():,}")

# 고유 그룹 수 확인
unique_groups = df_final[GROUP_KEYS].drop_duplicates().shape[0]
print(f"  고유 (상권, 업종) 조합 수: {unique_groups:,}개")

