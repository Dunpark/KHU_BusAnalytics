# -*- coding: utf-8 -*-
import pandas as pd

print("=" * 70)
print("1. 상주인구-상권 데이터셋 정밀 분석")
print("=" * 70)

# 상주인구-상권 파일 읽기
try:
    pop_df = pd.read_csv('Data_Raw_new/서울시 상권분석서비스(상주인구-상권).csv', encoding='utf-8-sig', low_memory=False)
except UnicodeDecodeError:
    pop_df = pd.read_csv('Data_Raw_new/서울시 상권분석서비스(상주인구-상권).csv', encoding='cp949', low_memory=False)

print(f"\n[기본 정보]")
print(f"  행 수: {len(pop_df):,}")
print(f"  컬럼 수: {len(pop_df.columns)}")

print(f"\n[컬럼 목록]")
for i, col in enumerate(pop_df.columns):
    print(f"  {i+1:2}. {col}")

print(f"\n[기준_년분기_코드 범위]")
quarters = sorted(pop_df['기준_년분기_코드'].unique())
print(f"  총 분기 수: {len(quarters)}개")
print(f"  분기 목록: {quarters}")

# 서비스_업종_코드 확인
print(f"\n[서비스_업종_코드 존재 여부]")
if '서비스_업종_코드' in pop_df.columns:
    print(f"  서비스_업종_코드 컬럼 있음")
else:
    print(f"  서비스_업종_코드 컬럼 없음 (상권 단위 데이터)")

print("\n" + "=" * 70)
print("2. 조합 비교 분석")
print("=" * 70)

# 현재 Merge된 데이터 로드
current_df = pd.read_csv('Merged_datasets/4개년추정매출_소득소비.csv', encoding='utf-8-sig', low_memory=False)
print(f"\n[현재 데이터: 4개년추정매출_소득소비.csv]")
print(f"  행 수: {len(current_df):,}")

KEY_COLS = ['기준_년분기_코드', '상권_구분_코드', '상권_코드']

# 현재 데이터에서 고유 조합
current_combos = set(current_df[KEY_COLS].drop_duplicates().apply(tuple, axis=1))
print(f"  고유 조합 수: {len(current_combos):,}개")

# 상주인구-상권에서 고유 조합
pop_combos = set(pop_df[KEY_COLS].drop_duplicates().apply(tuple, axis=1))
print(f"\n[상주인구-상권]")
print(f"  고유 조합 수: {len(pop_combos):,}개")

# 매칭 분석
common = current_combos.intersection(pop_combos)
only_in_current = current_combos - pop_combos
print(f"\n[매칭 결과]")
print(f"  공통 조합: {len(common):,}개")
print(f"  현재 데이터에만 있는 조합 (누락 예정): {len(only_in_current):,}개 ({len(only_in_current)/len(current_combos)*100:.1f}%)")

print("\n" + "=" * 70)
print("3. Inner Join 수행")
print("=" * 70)

# 상주인구-상권에서 중복 컬럼 제거
pop_cols_to_add = [c for c in pop_df.columns if c not in ['상권_구분_코드_명', '상권_코드_명']]
pop_df_clean = pop_df[pop_cols_to_add]
print(f"\n  조인 키: {KEY_COLS}")
print(f"  추가될 컬럼: {len(pop_df_clean.columns) - len(KEY_COLS)}개")

# Inner Join
merged_df = pd.merge(current_df, pop_df_clean, on=KEY_COLS, how='inner')

print(f"\n  [Merge 결과]")
print(f"  Merge 전: {len(current_df):,}행")
print(f"  Merge 후: {len(merged_df):,}행")
print(f"  이번 단계 누락: {len(current_df) - len(merged_df):,}행 ({(len(current_df) - len(merged_df))/len(current_df)*100:.2f}%)")

# 누락된 행 상세
if len(current_df) > len(merged_df):
    print("\n  [이번 단계 누락 상세]")
    missing_keys = only_in_current
    if missing_keys:
        missing_df = pd.DataFrame(list(missing_keys), columns=KEY_COLS)
        print("\n  분기별 누락:")
        for q, cnt in missing_df.groupby('기준_년분기_코드').size().sort_index().items():
            print(f"    {q}: {cnt}개 조합")
        print("\n  상권구분별 누락:")
        for t, cnt in missing_df.groupby('상권_구분_코드').size().items():
            print(f"    {t}: {cnt}개 조합")

# 결과 저장
print("\n" + "=" * 70)
print("4. 결과 저장")
print("=" * 70)

output_path = 'Merged_datasets/4개년추정매출_소득소비_상주인구.csv'
merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n  저장 완료: {output_path}")

# 원본 대비 비교
print("\n" + "=" * 70)
print("5. 원본 대비 최종 비교")
print("=" * 70)

original_rows = 290416  # 4개년추정매출_all_quarters.csv 원본 행 수
print(f"\n  [원본] 4개년추정매출_all_quarters.csv: {original_rows:,}행")
print(f"  [현재] 4개년추정매출_소득소비_상주인구.csv: {len(merged_df):,}행")
print(f"\n  총 누락된 행: {original_rows - len(merged_df):,}행")
print(f"  원본 대비 감소율: {(original_rows - len(merged_df))/original_rows*100:.2f}%")
print(f"  원본 대비 유지율: {len(merged_df)/original_rows*100:.2f}%")

# 추가된 컬럼
added_cols = [c for c in merged_df.columns if c not in current_df.columns]
print(f"\n  [추가된 상주인구 컬럼] ({len(added_cols)}개)")
for col in added_cols[:10]:
    print(f"    - {col}")
if len(added_cols) > 10:
    print(f"    ... 외 {len(added_cols)-10}개")

print(f"\n  총 컬럼 수: {len(merged_df.columns)}")
print(f"  총 결측치: {merged_df.isna().sum().sum():,}개")

