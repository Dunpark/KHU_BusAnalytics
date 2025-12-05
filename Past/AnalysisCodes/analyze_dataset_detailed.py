"""
통합 데이터셋 상세 분석 스크립트
Step 3: 연도별 추세, 분포, 상관관계 분석
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드
print("=" * 80)
print("📊 통합 데이터셋 상세 분석 (연도별 추세, 분포, 상관)")
print("=" * 80)

file_path = '../Merged_datasets/4개년_통합데이터_추정매출_상주인구_소득소비_길단위인구_점포_영역.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# 기준_년분기_코드에서 연도, 분기 추출
df['기준_년_코드'] = df['기준_년분기_코드'] // 10
df['기준_분기_코드'] = df['기준_년분기_코드'] % 10

print("\n" + "=" * 80)
print("1️⃣ 연도/분기별 데이터 분포")
print("=" * 80)

print("\n[연도별 데이터 수]")
year_counts = df['기준_년_코드'].value_counts().sort_index()
for year, count in year_counts.items():
    print(f"  {year}년: {count:,}건")

print("\n[분기별 데이터 수]")
quarter_counts = df['기준_분기_코드'].value_counts().sort_index()
for q, count in quarter_counts.items():
    print(f"  {q}분기: {count:,}건")

print("\n[연도-분기 조합별 데이터 수]")
yq_counts = df.groupby(['기준_년_코드', '기준_분기_코드']).size().unstack(fill_value=0)
print(yq_counts)

# 2. 상권 구분별 분포
print("\n" + "=" * 80)
print("2️⃣ 상권 구분별 분포")
print("=" * 80)

print("\n[상권 구분 코드별 분포]")
area_dist = df['상권_구분_코드_명'].value_counts()
for name, count in area_dist.items():
    pct = count / len(df) * 100
    print(f"  {name}: {count:,}건 ({pct:.1f}%)")

# 3. 업종별 분포 (상위 20개)
print("\n" + "=" * 80)
print("3️⃣ 서비스 업종별 분포 (상위 20개)")
print("=" * 80)

service_dist = df['서비스_업종_코드_명'].value_counts().head(20)
for name, count in service_dist.items():
    pct = count / len(df) * 100
    print(f"  {name}: {count:,}건 ({pct:.1f}%)")

# 4. 자치구별 분포
print("\n" + "=" * 80)
print("4️⃣ 자치구별 분포")
print("=" * 80)

gu_dist = df['자치구_코드_명'].value_counts().sort_values(ascending=False)
for name, count in gu_dist.items():
    pct = count / len(df) * 100
    print(f"  {name}: {count:,}건 ({pct:.1f}%)")

# 5. 연도별 주요 지표 추세
print("\n" + "=" * 80)
print("5️⃣ 연도별 주요 지표 추세")
print("=" * 80)

yearly_stats = df.groupby('기준_년_코드').agg({
    '당월_매출_금액': 'mean',
    '당월_매출_건수': 'mean',
    '점포_수': 'mean',
    '총_상주인구_수': 'mean',
    '총_유동인구_수': 'mean',
    '월_평균_소득_금액': 'mean',
    '폐업_점포_수': 'mean',
    '개업_점포_수': 'mean'
}).round(0)

print("\n[연도별 평균 당월 매출 금액]")
for year, val in yearly_stats['당월_매출_금액'].items():
    print(f"  {year}년: {val:,.0f}원")

print("\n[연도별 평균 점포 수]")
for year, val in yearly_stats['점포_수'].items():
    print(f"  {year}년: {val:,.1f}개")

print("\n[연도별 평균 총 유동인구 수]")
for year, val in yearly_stats['총_유동인구_수'].items():
    print(f"  {year}년: {val:,.0f}명")

print("\n[연도별 평균 폐업 점포 수]")
for year, val in yearly_stats['폐업_점포_수'].items():
    print(f"  {year}년: {val:,.2f}개")

print("\n[연도별 평균 개업 점포 수]")
for year, val in yearly_stats['개업_점포_수'].items():
    print(f"  {year}년: {val:,.2f}개")

# 6. 주요 변수 기술통계
print("\n" + "=" * 80)
print("6️⃣ 주요 변수 기술통계")
print("=" * 80)

key_vars = ['당월_매출_금액', '점포_수', '총_상주인구_수', '총_유동인구_수', 
            '월_평균_소득_금액', '지출_총금액', '영역_면적']

stats_df = df[key_vars].describe()
print("\n[주요 변수 기술통계]")
print(stats_df.round(2).to_string())

# 왜도(Skewness) 계산
print("\n[주요 변수 왜도(Skewness) - 0에 가까울수록 정규분포]")
for var in key_vars:
    skew = df[var].skew()
    print(f"  {var}: {skew:.2f}")

# 7. 주요 변수 간 상관관계
print("\n" + "=" * 80)
print("7️⃣ 주요 변수 간 상관관계 (매출과의 상관)")
print("=" * 80)

corr_vars = ['당월_매출_금액', '점포_수', '총_상주인구_수', '총_유동인구_수', 
             '월_평균_소득_금액', '지출_총금액', '영역_면적', '유사_업종_점포_수',
             '프랜차이즈_점포_수', '개업_점포_수', '폐업_점포_수']

corr_with_sales = df[corr_vars].corr()['당월_매출_금액'].sort_values(ascending=False)
print("\n[당월_매출_금액과의 상관계수]")
for var, corr in corr_with_sales.items():
    if var != '당월_매출_금액':
        print(f"  {var}: {corr:.3f}")

# 8. 업종별 평균 매출 (상위/하위 10개)
print("\n" + "=" * 80)
print("8️⃣ 업종별 평균 매출")
print("=" * 80)

service_sales = df.groupby('서비스_업종_코드_명')['당월_매출_금액'].mean().sort_values(ascending=False)

print("\n[고매출 업종 TOP 10]")
for i, (name, val) in enumerate(service_sales.head(10).items(), 1):
    print(f"  {i:2d}. {name}: {val:,.0f}원")

print("\n[저매출 업종 TOP 10]")
for i, (name, val) in enumerate(service_sales.tail(10).items(), 1):
    print(f"  {i:2d}. {name}: {val:,.0f}원")

# 9. 상권 유형별 매출 차이
print("\n" + "=" * 80)
print("9️⃣ 상권 유형별 평균 매출")
print("=" * 80)

area_sales = df.groupby('상권_구분_코드_명').agg({
    '당월_매출_금액': 'mean',
    '점포_수': 'mean',
    '총_유동인구_수': 'mean'
}).round(0)

print(area_sales.to_string())

# 10. 매출 분포 (분위수 기반)
print("\n" + "=" * 80)
print("🔟 매출 분포 분석 (분위수 기반)")
print("=" * 80)

quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
print("\n[당월_매출_금액 분위수]")
for q in quantiles:
    val = df['당월_매출_금액'].quantile(q)
    print(f"  {int(q*100):3d}%: {val:,.0f}원")

# 11. 연도별 업종별 변화 (대표 업종)
print("\n" + "=" * 80)
print("1️⃣1️⃣ 연도별 대표 업종 매출 변화")
print("=" * 80)

# 대표 업종 선정 (데이터 많은 업종)
top_services = df['서비스_업종_코드_명'].value_counts().head(5).index.tolist()

for service in top_services:
    service_df = df[df['서비스_업종_코드_명'] == service]
    yearly_mean = service_df.groupby('기준_년_코드')['당월_매출_금액'].mean()
    
    print(f"\n[{service}]")
    for year, val in yearly_mean.items():
        print(f"  {year}년: {val:,.0f}원")

# 12. 폐업률 높은 업종/상권
print("\n" + "=" * 80)
print("1️⃣2️⃣ 업종별 평균 폐업 점포 수")
print("=" * 80)

closure_by_service = df.groupby('서비스_업종_코드_명')['폐업_점포_수'].mean().sort_values(ascending=False)

print("\n[폐업 점포 수 높은 업종 TOP 10]")
for i, (name, val) in enumerate(closure_by_service.head(10).items(), 1):
    print(f"  {i:2d}. {name}: {val:.2f}개")

print("\n" + "=" * 80)
print("✅ 상세 분석 완료!")
print("=" * 80)

