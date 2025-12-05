"""
통합 데이터셋 구조 및 특성 분석 스크립트
Step 3: 데이터 정밀 이해를 위한 EDA
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드
print("=" * 80)
print("📊 통합 데이터셋 구조 및 특성 분석")
print("=" * 80)

file_path = '../Merged_datasets/4개년_통합데이터_추정매출_상주인구_소득소비_길단위인구_점포_영역.csv'

print("\n[1] 데이터 로딩 중...")
# 여러 인코딩 시도
encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin1']
for enc in encodings:
    try:
        df = pd.read_csv(file_path, encoding=enc)
        print(f"✅ 인코딩 '{enc}'로 로딩 성공!")
        break
    except Exception as e:
        print(f"❌ 인코딩 '{enc}' 실패: {str(e)[:50]}")
print(f"✅ 로딩 완료!")

# 1. 기본 정보
print("\n" + "=" * 80)
print("1️⃣ 데이터 기본 정보")
print("=" * 80)
print(f"• 전체 행(row) 수: {df.shape[0]:,}")
print(f"• 전체 열(column) 수: {df.shape[1]}")
print(f"• 메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 2. 컬럼 목록 및 데이터 타입
print("\n" + "=" * 80)
print("2️⃣ 컬럼 목록 및 데이터 타입")
print("=" * 80)

# 컬럼을 카테고리별로 분류
print("\n[전체 컬럼 목록]")
for i, col in enumerate(df.columns):
    dtype = df[col].dtype
    null_count = df[col].isnull().sum()
    null_pct = null_count / len(df) * 100
    print(f"{i+1:3d}. {col:<50} | {str(dtype):<10} | 결측: {null_count:,} ({null_pct:.1f}%)")

# 3. 키 변수 분석
print("\n" + "=" * 80)
print("3️⃣ 키 변수 분석 (기준 연/분기, 상권, 업종)")
print("=" * 80)

# 연도/분기 분포
if '기준_년_코드' in df.columns:
    print("\n[연도별 데이터 분포]")
    print(df['기준_년_코드'].value_counts().sort_index())
    
if '기준_분기_코드' in df.columns:
    print("\n[분기별 데이터 분포]")
    print(df['기준_분기_코드'].value_counts().sort_index())

# 연도-분기 조합
if '기준_년_코드' in df.columns and '기준_분기_코드' in df.columns:
    print("\n[연도-분기 조합별 데이터 수]")
    year_quarter = df.groupby(['기준_년_코드', '기준_분기_코드']).size()
    print(year_quarter)

# 상권 코드
if '상권_코드' in df.columns:
    print(f"\n[상권 코드]")
    print(f"• 고유 상권 수: {df['상권_코드'].nunique():,}")
    print(f"• 상권 코드 예시: {df['상권_코드'].head(5).tolist()}")

# 서비스 업종 코드
service_cols = [col for col in df.columns if '서비스_업종' in col or '업종' in col]
if service_cols:
    print(f"\n[업종 관련 컬럼]")
    for col in service_cols[:3]:
        print(f"• {col}: 고유값 {df[col].nunique():,}개")
        if df[col].nunique() < 20:
            print(f"  값 분포: {df[col].value_counts().head(10).to_dict()}")

# 4. 주요 수치형 변수 통계
print("\n" + "=" * 80)
print("4️⃣ 주요 수치형 변수 기술통계")
print("=" * 80)

# 매출 관련 변수
sales_cols = [col for col in df.columns if '매출' in col]
print(f"\n[매출 관련 컬럼 수: {len(sales_cols)}개]")
if sales_cols:
    print("매출 컬럼 목록:", sales_cols[:10], "..." if len(sales_cols) > 10 else "")
    
    # 주요 매출 변수 통계
    key_sales = [col for col in sales_cols if '분기당' in col or '금액' in col][:5]
    if key_sales:
        print("\n[주요 매출 변수 통계]")
        print(df[key_sales].describe().round(0))

# 인구 관련 변수  
pop_cols = [col for col in df.columns if '인구' in col]
print(f"\n[인구 관련 컬럼 수: {len(pop_cols)}개]")
if pop_cols:
    print("인구 컬럼 목록:", pop_cols[:10], "..." if len(pop_cols) > 10 else "")

# 소득/소비 관련 변수
income_cols = [col for col in df.columns if '소득' in col or '소비' in col or '지출' in col]
print(f"\n[소득/소비 관련 컬럼 수: {len(income_cols)}개]")
if income_cols:
    print("소득/소비 컬럼 목록:", income_cols[:10], "..." if len(income_cols) > 10 else "")

# 점포 관련 변수
store_cols = [col for col in df.columns if '점포' in col or '개업' in col or '폐업' in col]
print(f"\n[점포 관련 컬럼 수: {len(store_cols)}개]")
if store_cols:
    print("점포 컬럼 목록:", store_cols[:10], "..." if len(store_cols) > 10 else "")

# 영역 관련 변수
area_cols = [col for col in df.columns if '영역' in col or '면적' in col or '좌표' in col]
print(f"\n[영역 관련 컬럼 수: {len(area_cols)}개]")
if area_cols:
    print("영역 컬럼 목록:", area_cols[:10], "..." if len(area_cols) > 10 else "")

# 5. 결측치 분석
print("\n" + "=" * 80)
print("5️⃣ 결측치 현황 분석")
print("=" * 80)

null_counts = df.isnull().sum()
null_cols = null_counts[null_counts > 0].sort_values(ascending=False)

print(f"\n• 결측치가 있는 컬럼 수: {len(null_cols)}")
print(f"• 결측치가 없는 컬럼 수: {len(df.columns) - len(null_cols)}")

if len(null_cols) > 0:
    print("\n[결측치가 많은 상위 20개 컬럼]")
    for col, count in null_cols.head(20).items():
        pct = count / len(df) * 100
        print(f"  • {col}: {count:,} ({pct:.1f}%)")

# 6. 데이터 품질 이슈
print("\n" + "=" * 80)
print("6️⃣ 데이터 품질 이슈 점검")
print("=" * 80)

# 중복 행 체크
dup_count = df.duplicated().sum()
print(f"\n• 완전 중복 행 수: {dup_count:,}")

# 키 조합 중복 체크
if '기준_년_코드' in df.columns and '기준_분기_코드' in df.columns and '상권_코드' in df.columns:
    key_cols = ['기준_년_코드', '기준_분기_코드', '상권_코드']
    if '서비스_업종_코드' in df.columns:
        key_cols.append('서비스_업종_코드')
    
    key_dup = df.duplicated(subset=key_cols).sum()
    print(f"• 키 조합({', '.join(key_cols)}) 중복: {key_dup:,}")

# 음수 값 체크 (매출, 인구 등)
print("\n[음수 값 존재 여부 - 주요 수치형 컬럼]")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols[:20]:
    neg_count = (df[col] < 0).sum()
    if neg_count > 0:
        print(f"  • {col}: {neg_count:,}개 음수값")

# 7. 연도별 주요 지표 추세
print("\n" + "=" * 80)
print("7️⃣ 연도별 주요 지표 추세")
print("=" * 80)

if '기준_년_코드' in df.columns:
    # 매출 추세
    sales_trend_cols = [col for col in df.columns if '분기당_매출_금액' in col]
    if sales_trend_cols:
        print("\n[연도별 평균 매출 추세]")
        yearly_sales = df.groupby('기준_년_코드')[sales_trend_cols[0]].mean()
        for year, val in yearly_sales.items():
            print(f"  {year}년: {val:,.0f}")
    
    # 점포수 추세
    if '점포_수' in df.columns:
        print("\n[연도별 평균 점포 수 추세]")
        yearly_store = df.groupby('기준_년_코드')['점포_수'].mean()
        for year, val in yearly_store.items():
            print(f"  {year}년: {val:,.1f}")

# 8. 샘플 데이터 출력
print("\n" + "=" * 80)
print("8️⃣ 샘플 데이터 (처음 3행)")
print("=" * 80)
print(df.head(3).T)  # Transpose for better readability

print("\n" + "=" * 80)
print("✅ 분석 완료!")
print("=" * 80)

