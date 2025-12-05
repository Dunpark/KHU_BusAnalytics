# -*- coding: utf-8 -*-
"""
4개년 추정매출-상권 데이터에 소득소비-상권, 상주인구-상권 데이터를 병합하는 코드
- LEFT JOIN 방식 사용
- 서비스_업종코드 유무에 따른 동적 조인 키 설정
- 결측치 기반 그룹 필터링
"""

import pandas as pd

# =============================================================================
# 1. 데이터 로드
# =============================================================================
print("=" * 70)
print("[1/5] 데이터 로드")
print("=" * 70)

# 추정매출-상권 데이터 (4개년 통합)
df_sales = pd.read_csv(
    'Merged_datasets/4개년추정매출_all_quarters.csv',
    encoding='utf-8-sig',
    low_memory=False
)
print(f"df_sales 로드 완료: {df_sales.shape}")

# 소득소비-상권 데이터
try:
    df_income = pd.read_csv(
        'Data_Raw_new/서울시 상권분석서비스(소득소비-상권).csv',
        encoding='utf-8-sig',
        low_memory=False
    )
except UnicodeDecodeError:
    df_income = pd.read_csv(
        'Data_Raw_new/서울시 상권분석서비스(소득소비-상권).csv',
        encoding='cp949',
        low_memory=False
    )
print(f"df_income 로드 완료: {df_income.shape}")

# 상주인구-상권 데이터
try:
    df_resident = pd.read_csv(
        'Data_Raw_new/서울시 상권분석서비스(상주인구-상권).csv',
        encoding='utf-8-sig',
        low_memory=False
    )
except UnicodeDecodeError:
    df_resident = pd.read_csv(
        'Data_Raw_new/서울시 상권분석서비스(상주인구-상권).csv',
        encoding='cp949',
        low_memory=False
    )
print(f"df_resident 로드 완료: {df_resident.shape}")

# =============================================================================
# 2. 조인 키 동적 설정
# =============================================================================
print("\n" + "=" * 70)
print("[2/5] 조인 키 동적 설정")
print("=" * 70)

# 기본 조인 키 (3개 컬럼)
BASE_JOIN_KEYS = ["기준_년분기_코드", "상권_코드", "서비스_업종_코드"]
# 축소된 조인 키 (2개 컬럼) - 서비스_업종_코드가 없는 경우
REDUCED_JOIN_KEYS = ["기준_년분기_코드", "상권_코드"]

# df_income에 서비스_업종_코드가 있는지 확인
if "서비스_업종_코드" in df_income.columns:
    join_keys_income = BASE_JOIN_KEYS
    print(f"df_income 조인 키: {join_keys_income} (서비스_업종_코드 있음)")
else:
    join_keys_income = REDUCED_JOIN_KEYS
    print(f"df_income 조인 키: {join_keys_income} (서비스_업종_코드 없음)")

# df_resident에 서비스_업종_코드가 있는지 확인
if "서비스_업종_코드" in df_resident.columns:
    join_keys_resident = BASE_JOIN_KEYS
    print(f"df_resident 조인 키: {join_keys_resident} (서비스_업종_코드 있음)")
else:
    join_keys_resident = REDUCED_JOIN_KEYS
    print(f"df_resident 조인 키: {join_keys_resident} (서비스_업종_코드 없음)")

# =============================================================================
# 3. 소득소비-상권 데이터 Merge (df_sales + df_income → df_tmp)
# =============================================================================
print("\n" + "=" * 70)
print("[3/5] 소득소비-상권 데이터 Merge")
print("=" * 70)

# df_income에서 조인 키를 제외한 컬럼들에 suffix 추가를 위해 컬럼 이름 확인
# 조인 키와 중복되는 컬럼(상권_코드_명, 상권_구분_코드_명 등)은 제거하여 충돌 방지
income_cols_to_drop = [col for col in df_income.columns 
                       if col not in join_keys_income and col in df_sales.columns]
print(f"df_income에서 제거할 중복 컬럼: {income_cols_to_drop}")

# 중복 컬럼 제거
df_income_clean = df_income.drop(columns=income_cols_to_drop, errors='ignore')

# 소득소비 관련 컬럼 리스트 저장 (나중에 결측치 필터링에 사용)
income_data_cols = [col for col in df_income_clean.columns if col not in join_keys_income]
print(f"소득소비 관련 컬럼 수: {len(income_data_cols)}개")
print(f"소득소비 관련 컬럼: {income_data_cols}")

# LEFT JOIN 수행
# df_sales를 기준으로, df_income_clean을 조인
df_tmp = pd.merge(
    df_sales,
    df_income_clean,
    on=join_keys_income,
    how='left',  # LEFT JOIN: df_sales의 모든 행 유지
    suffixes=('', '_income')  # 혹시 모를 컬럼명 충돌 대비
)

print(f"\nMerge 결과:")
print(f"  df_sales: {df_sales.shape[0]:,}행")
print(f"  df_tmp (merge 후): {df_tmp.shape[0]:,}행")
print(f"  컬럼 수 변화: {df_sales.shape[1]} → {df_tmp.shape[1]}")

# =============================================================================
# 4. 상주인구-상권 데이터 Merge (df_tmp + df_resident → df_merged)
# =============================================================================
print("\n" + "=" * 70)
print("[4/5] 상주인구-상권 데이터 Merge")
print("=" * 70)

# df_resident에서 조인 키를 제외한 컬럼들 중 중복되는 것 제거
resident_cols_to_drop = [col for col in df_resident.columns 
                         if col not in join_keys_resident and col in df_tmp.columns]
print(f"df_resident에서 제거할 중복 컬럼: {resident_cols_to_drop}")

# 중복 컬럼 제거
df_resident_clean = df_resident.drop(columns=resident_cols_to_drop, errors='ignore')

# 상주인구 관련 컬럼 리스트 저장 (나중에 결측치 필터링에 사용)
resident_data_cols = [col for col in df_resident_clean.columns if col not in join_keys_resident]
print(f"상주인구 관련 컬럼 수: {len(resident_data_cols)}개")
print(f"상주인구 관련 컬럼: {resident_data_cols[:10]}...")  # 처음 10개만 출력

# LEFT JOIN 수행
df_merged = pd.merge(
    df_tmp,
    df_resident_clean,
    on=join_keys_resident,
    how='left',  # LEFT JOIN: df_tmp의 모든 행 유지
    suffixes=('', '_resident')  # 혹시 모를 컬럼명 충돌 대비
)

print(f"\nMerge 결과:")
print(f"  df_tmp: {df_tmp.shape[0]:,}행")
print(f"  df_merged (merge 후): {df_merged.shape[0]:,}행")
print(f"  컬럼 수 변화: {df_tmp.shape[1]} → {df_merged.shape[1]}")

# =============================================================================
# 5. 결측치 기반 필터링 (상권_코드, 서비스_업종_코드 그룹별)
# =============================================================================
print("\n" + "=" * 70)
print("[5/5] 결측치 기반 필터링")
print("=" * 70)

# 그룹화 키: 상권_코드, 서비스_업종_코드
GROUP_KEYS = ["상권_코드", "서비스_업종_코드"]

# 결측치 확인 대상 컬럼: 소득소비 + 상주인구 관련 컬럼
# (Merge 과정에서 컬럼명이 변경되었을 수 있으므로 실제 df_merged의 컬럼에서 찾기)
check_cols_income = [col for col in df_merged.columns if col in income_data_cols]
check_cols_resident = [col for col in df_merged.columns if col in resident_data_cols]
check_cols = check_cols_income + check_cols_resident

print(f"결측치 확인 대상 컬럼 수: {len(check_cols)}개")
print(f"  - 소득소비 관련: {len(check_cols_income)}개")
print(f"  - 상주인구 관련: {len(check_cols_resident)}개")

# 각 그룹(상권_코드, 서비스_업종_코드)별로 결측치 존재 여부 확인
# 그룹 내 check_cols 중 하나라도 NaN이 있으면 해당 그룹 전체 제거
print("\n결측치 검사 중...")

# 방법: 각 행에서 check_cols 중 하나라도 NaN이면 True
df_merged['has_missing'] = df_merged[check_cols].isna().any(axis=1)

# 그룹별로 has_missing이 하나라도 True인 그룹 찾기
# transform으로 각 그룹의 has_missing 합계를 모든 행에 전파
df_merged['group_has_missing'] = df_merged.groupby(GROUP_KEYS)['has_missing'].transform('sum')

# group_has_missing > 0이면 해당 그룹에 결측치가 있는 것
# 이런 그룹들을 제거
df_final = df_merged[df_merged['group_has_missing'] == 0].copy()

# 임시 컬럼 삭제
df_final = df_final.drop(columns=['has_missing', 'group_has_missing'])

# 결과 출력
removed_rows = df_merged.shape[0] - df_final.shape[0]
removed_groups = df_merged[df_merged['group_has_missing'] > 0][GROUP_KEYS].drop_duplicates().shape[0]

print(f"\n필터링 결과:")
print(f"  df_merged (필터링 전): {df_merged.shape[0]:,}행")
print(f"  df_final (필터링 후): {df_final.shape[0]:,}행")
print(f"  제거된 행 수: {removed_rows:,}행 ({removed_rows/df_merged.shape[0]*100:.2f}%)")
print(f"  제거된 그룹 수: {removed_groups:,}개")

# =============================================================================
# 6. 최종 결과 확인
# =============================================================================
print("\n" + "=" * 70)
print("최종 결과 요약")
print("=" * 70)

print(f"\n[df_final 정보]")
print(f"  행 수: {df_final.shape[0]:,}")
print(f"  컬럼 수: {df_final.shape[1]}")
print(f"  결측치 수: {df_final.isna().sum().sum():,}")

# 고유 그룹 수 확인
unique_groups = df_final[GROUP_KEYS].drop_duplicates().shape[0]
print(f"  고유 (상권, 업종) 조합 수: {unique_groups:,}개")

# 분기 수 확인
unique_quarters = df_final['기준_년분기_코드'].nunique()
print(f"  고유 분기 수: {unique_quarters}개")

print("\n[df_final.head() 출력]")
print(df_final.head())

# =============================================================================
# 7. 결과 저장
# =============================================================================
output_path = 'Merged_datasets/4개년_통합데이터_final.csv'
df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n저장 완료: {output_path}")

