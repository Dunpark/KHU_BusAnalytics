"""
================================================================================
📊 서울시 상권 유형화 분석 (K-means 군집 분석)
================================================================================
- 데이터: 한식음식점 4개년 통합 데이터
- 분석 목적: 상권을 데이터 기반으로 유형화 (성장형/안정형/쇠퇴형 등)
- 특징: 시계열 기반 파생변수 포함, PCA 차원 축소 적용

실행: python ClusteringAnalysis.py
================================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# 경고 무시
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 출력 폴더 생성
OUTPUT_DIR = "ClusteringResults"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """
    1단계: 데이터 로드
    - 한식음식점 4개년 통합 데이터 로드
    """
    print("=" * 80)
    print("📁 1단계: 데이터 로드")
    print("=" * 80)
    
    data_path = "BrainstormingAnalytics/4개년_한식음식점_통합데이터_추정매출_상주인구_소득소비_길단위인구_점포_영역.csv"
    
    df = pd.read_csv(data_path, encoding='utf-8')
    
    print(f"✅ 데이터 로드 완료")
    print(f"   - 행 수: {len(df):,}")
    print(f"   - 열 수: {df.shape[1]}")
    print(f"   - 기간: {df['기준_년분기_코드'].min()} ~ {df['기준_년분기_코드'].max()}")
    print(f"   - 고유 상권 수: {df['상권_코드'].nunique():,}")
    
    return df


def extract_time_features(df):
    """
    2단계: 시계열 기반 파생변수 생성
    - 매출 성장률, 변동성, 추세
    """
    print("\n" + "=" * 80)
    print("📈 2단계: 시계열 기반 파생변수 생성")
    print("=" * 80)
    
    # 연도 추출
    df['연도'] = df['기준_년분기_코드'] // 10
    df['분기'] = df['기준_년분기_코드'] % 10
    
    # 상권별 집계를 위한 빈 DataFrame
    time_features = []
    
    for sangkwon_code in df['상권_코드'].unique():
        sangkwon_df = df[df['상권_코드'] == sangkwon_code].sort_values('기준_년분기_코드')
        
        # 기본 정보
        sangkwon_name = sangkwon_df['상권_코드_명'].iloc[0]
        area_type = sangkwon_df['상권_구분_코드_명'].iloc[0]
        
        # 매출 데이터
        sales = sangkwon_df['당월_매출_금액'].values
        quarters = sangkwon_df['기준_년분기_코드'].values
        
        # 연도별 평균 매출
        yearly_sales = sangkwon_df.groupby('연도')['당월_매출_금액'].mean()
        
        # ----- 시계열 파생변수 -----
        
        # 1. 매출 성장률 (첫해 대비 마지막해)
        if 2021 in yearly_sales.index and 2024 in yearly_sales.index:
            if yearly_sales[2021] > 0:
                growth_rate = (yearly_sales[2024] - yearly_sales[2021]) / yearly_sales[2021] * 100
            else:
                growth_rate = np.nan
        else:
            growth_rate = np.nan
        
        # 2. 매출 변동성 (변동계수 = 표준편차/평균)
        if sales.mean() > 0:
            sales_cv = sales.std() / sales.mean() * 100
        else:
            sales_cv = np.nan
        
        # 3. 매출 추세 (선형 회귀 기울기)
        if len(sales) > 1:
            x = np.arange(len(sales))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, sales)
            # 기울기를 평균 매출 대비 % 변화로 정규화
            if sales.mean() > 0:
                sales_trend = slope / sales.mean() * 100
            else:
                sales_trend = np.nan
        else:
            sales_trend = np.nan
        
        # 4. 최근 분기 대비 변화율 (최근 4분기 평균 vs 이전 4분기 평균)
        if len(sales) >= 8:
            recent_avg = sales[-4:].mean()
            previous_avg = sales[-8:-4].mean()
            if previous_avg > 0:
                recent_change = (recent_avg - previous_avg) / previous_avg * 100
            else:
                recent_change = np.nan
        else:
            recent_change = np.nan
        
        # ----- 정적 변수 (최신 연도 평균) -----
        latest_year = sangkwon_df[sangkwon_df['연도'] == 2024]
        if len(latest_year) == 0:
            latest_year = sangkwon_df[sangkwon_df['연도'] == sangkwon_df['연도'].max()]
        
        avg_sales = latest_year['당월_매출_금액'].mean()
        avg_stores = latest_year['점포_수'].mean()
        avg_floating_pop = latest_year['총_유동인구_수'].mean()
        avg_resident_pop = latest_year['총_상주인구_수'].mean()
        avg_income = latest_year['월_평균_소득_금액'].mean()
        avg_area = latest_year['영역_면적'].mean()
        
        # 추가 정적 변수
        avg_franchise = latest_year['프랜차이즈_점포_수'].mean() if '프랜차이즈_점포_수' in latest_year.columns else 0
        avg_opening_rate = latest_year['개업_율'].mean() if '개업_율' in latest_year.columns else 0
        avg_closing_rate = latest_year['폐업_률'].mean() if '폐업_률' in latest_year.columns else 0
        
        # 시간대별 매출 비율 (동적 패턴)
        total_sales = latest_year['당월_매출_금액'].sum()
        if total_sales > 0:
            lunch_ratio = latest_year['시간대_11~14_매출_금액'].sum() / total_sales * 100
            dinner_ratio = latest_year['시간대_17~21_매출_금액'].sum() / total_sales * 100
            night_ratio = latest_year['시간대_21~24_매출_금액'].sum() / total_sales * 100
        else:
            lunch_ratio = dinner_ratio = night_ratio = np.nan
        
        # 주중/주말 매출 비율
        weekday_sales = latest_year['주중_매출_금액'].sum()
        weekend_sales = latest_year['주말_매출_금액'].sum()
        if (weekday_sales + weekend_sales) > 0:
            weekend_ratio = weekend_sales / (weekday_sales + weekend_sales) * 100
        else:
            weekend_ratio = np.nan
        
        time_features.append({
            '상권_코드': sangkwon_code,
            '상권_코드_명': sangkwon_name,
            '상권_구분_코드_명': area_type,
            
            # 시계열 파생변수
            '매출_성장률': growth_rate,
            '매출_변동성': sales_cv,
            '매출_추세': sales_trend,
            '최근_변화율': recent_change,
            
            # 정적 변수
            '평균_매출': avg_sales,
            '점포_수': avg_stores,
            '총_유동인구_수': avg_floating_pop,
            '총_상주인구_수': avg_resident_pop,
            '월_평균_소득_금액': avg_income,
            '영역_면적': avg_area,
            '프랜차이즈_점포_수': avg_franchise,
            '개업_율': avg_opening_rate,
            '폐업_률': avg_closing_rate,
            
            # 시간대/요일 패턴
            '점심_매출_비율': lunch_ratio,
            '저녁_매출_비율': dinner_ratio,
            '야간_매출_비율': night_ratio,
            '주말_매출_비율': weekend_ratio,
        })
    
    features_df = pd.DataFrame(time_features)
    
    print(f"✅ 시계열 파생변수 생성 완료")
    print(f"   - 상권 수: {len(features_df):,}")
    print(f"   - 변수 수: {features_df.shape[1]}")
    
    # 결측치 현황
    print("\n📊 파생변수 결측치 현황:")
    missing = features_df.isnull().sum()
    for col in missing[missing > 0].index:
        print(f"   - {col}: {missing[col]} ({missing[col]/len(features_df)*100:.1f}%)")
    
    return features_df


def preprocess_features(features_df):
    """
    3단계: 변수 선택 및 전처리
    - 분석에 사용할 변수 선택
    - 결측치 처리
    - 이상치 처리
    """
    print("\n" + "=" * 80)
    print("🔧 3단계: 변수 선택 및 전처리")
    print("=" * 80)
    
    # 분석에 사용할 변수 선택
    feature_cols = [
        # 시계열 파생변수
        '매출_성장률',
        '매출_변동성',
        '매출_추세',
        
        # 정적 변수 (로그 변환 대상)
        '평균_매출',
        '점포_수',
        '총_유동인구_수',
        '총_상주인구_수',
        '월_평균_소득_금액',
        '영역_면적',
        
        # 시간대/요일 패턴
        '점심_매출_비율',
        '저녁_매출_비율',
        '주말_매출_비율',
    ]
    
    df_analysis = features_df[['상권_코드', '상권_코드_명', '상권_구분_코드_명'] + feature_cols].copy()
    
    # 결측치 처리 (중앙값 대체)
    print("\n📍 결측치 처리:")
    for col in feature_cols:
        n_missing = df_analysis[col].isnull().sum()
        if n_missing > 0:
            median_val = df_analysis[col].median()
            df_analysis[col].fillna(median_val, inplace=True)
            print(f"   - {col}: {n_missing}개 → 중앙값({median_val:.2f})으로 대체")
    
    # 로그 변환 (우편향 변수)
    log_cols = ['평균_매출', '점포_수', '총_유동인구_수', '총_상주인구_수', '영역_면적']
    print("\n📍 로그 변환:")
    for col in log_cols:
        original_skew = df_analysis[col].skew()
        df_analysis[f'{col}_log'] = np.log1p(df_analysis[col])
        new_skew = df_analysis[f'{col}_log'].skew()
        print(f"   - {col}: 왜도 {original_skew:.2f} → {new_skew:.2f}")
    
    # 최종 분석 변수 선택
    final_feature_cols = [
        '매출_성장률',
        '매출_변동성',
        '매출_추세',
        '평균_매출_log',
        '점포_수_log',
        '총_유동인구_수_log',
        '총_상주인구_수_log',
        '월_평균_소득_금액',
        '영역_면적_log',
        '점심_매출_비율',
        '저녁_매출_비율',
        '주말_매출_비율',
    ]
    
    print(f"\n✅ 전처리 완료")
    print(f"   - 최종 분석 변수: {len(final_feature_cols)}개")
    for col in final_feature_cols:
        print(f"      • {col}")
    
    return df_analysis, final_feature_cols


def standardize_and_pca(df_analysis, feature_cols):
    """
    4단계: 표준화 및 PCA 차원 축소
    - z-score 표준화
    - PCA로 핵심 요인 추출 (선행연구 Factor Analysis 참고)
    """
    print("\n" + "=" * 80)
    print("📐 4단계: 표준화 및 PCA 차원 축소")
    print("=" * 80)
    
    X = df_analysis[feature_cols].values
    
    # z-score 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("✅ z-score 표준화 완료")
    
    # PCA 수행 (설명력 95% 이상)
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    # 누적 설명력 계산
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumsum_var >= 0.90) + 1  # 90% 설명력
    
    print(f"\n📊 PCA 분석 결과:")
    print(f"   - 원본 변수 수: {len(feature_cols)}")
    print(f"   - 90% 설명력 달성 주성분 수: {n_components}")
    
    # 주성분별 설명력
    print("\n   주성분별 설명력:")
    for i, var in enumerate(pca_full.explained_variance_ratio_[:n_components+2]):
        cum_var = cumsum_var[i]
        print(f"      PC{i+1}: {var*100:.1f}% (누적: {cum_var*100:.1f}%)")
    
    # 최종 PCA 적용
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # 주성분 해석 (로딩)
    print("\n📊 주성분 해석 (주요 로딩):")
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=feature_cols
    )
    
    for i in range(min(3, n_components)):
        pc = f'PC{i+1}'
        print(f"\n   {pc} (설명력: {pca.explained_variance_ratio_[i]*100:.1f}%):")
        top_loadings = loadings[pc].abs().sort_values(ascending=False).head(3)
        for var in top_loadings.index:
            val = loadings.loc[var, pc]
            print(f"      • {var}: {val:.3f}")
    
    # 시각화: 설명력 스크리 플롯
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 개별 설명력
    axes[0].bar(range(1, len(pca_full.explained_variance_ratio_)+1), 
                pca_full.explained_variance_ratio_ * 100, color='steelblue', alpha=0.7)
    axes[0].axhline(y=10, color='red', linestyle='--', label='10% 기준선')
    axes[0].set_xlabel('주성분')
    axes[0].set_ylabel('설명력 (%)')
    axes[0].set_title('주성분별 설명력 (Scree Plot)')
    axes[0].legend()
    
    # 누적 설명력
    axes[1].plot(range(1, len(cumsum_var)+1), cumsum_var * 100, 'o-', color='steelblue')
    axes[1].axhline(y=90, color='red', linestyle='--', label='90% 기준선')
    axes[1].axvline(x=n_components, color='green', linestyle='--', label=f'선택 주성분 ({n_components}개)')
    axes[1].set_xlabel('주성분 수')
    axes[1].set_ylabel('누적 설명력 (%)')
    axes[1].set_title('누적 설명력')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_pca_scree_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ PCA 차원 축소 완료: {len(feature_cols)}개 → {n_components}개 주성분")
    
    return X_scaled, X_pca, pca, scaler, loadings


def find_optimal_k(X, max_k=10):
    """
    5-1단계: 최적 k 탐색
    - Elbow Method
    - Silhouette Score
    """
    print("\n" + "=" * 80)
    print("🔍 5-1단계: 최적 k 탐색")
    print("=" * 80)
    
    inertias = []
    silhouettes = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, kmeans.labels_))
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow Plot
    axes[0].plot(K_range, inertias, 'o-', color='steelblue', linewidth=2, markersize=8)
    axes[0].set_xlabel('클러스터 수 (k)')
    axes[0].set_ylabel('Inertia (Within-cluster Sum of Squares)')
    axes[0].set_title('Elbow Method')
    axes[0].set_xticks(list(K_range))
    
    # Silhouette Score
    axes[1].plot(K_range, silhouettes, 'o-', color='coral', linewidth=2, markersize=8)
    axes[1].set_xlabel('클러스터 수 (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score by k')
    axes[1].set_xticks(list(K_range))
    axes[1].axhline(y=0.3, color='red', linestyle='--', label='최소 기준 (0.3)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_optimal_k_search.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 최적 k 결정 (실루엣 스코어 기준)
    best_k = K_range[np.argmax(silhouettes)]
    best_silhouette = max(silhouettes)
    
    print("📊 k별 성능:")
    for k, (inertia, sil) in enumerate(zip(inertias, silhouettes), start=2):
        marker = " ★" if k == best_k else ""
        print(f"   k={k}: Inertia={inertia:.0f}, Silhouette={sil:.3f}{marker}")
    
    print(f"\n✅ 최적 k 결정: {best_k} (Silhouette Score: {best_silhouette:.3f})")
    
    return best_k, silhouettes


def perform_kmeans(X, k):
    """
    5-2단계: K-means 클러스터링 수행
    """
    print("\n" + "=" * 80)
    print(f"🎯 5-2단계: K-means 클러스터링 (k={k})")
    print("=" * 80)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # 클러스터별 개수
    unique, counts = np.unique(labels, return_counts=True)
    
    print("✅ 클러스터링 완료")
    print("\n📊 클러스터별 상권 수:")
    for cluster, count in zip(unique, counts):
        print(f"   - 클러스터 {cluster}: {count}개 ({count/len(labels)*100:.1f}%)")
    
    # 실루엣 스코어
    overall_silhouette = silhouette_score(X, labels)
    sample_silhouettes = silhouette_samples(X, labels)
    
    print(f"\n📊 실루엣 스코어:")
    print(f"   - 전체: {overall_silhouette:.3f}")
    for cluster in unique:
        cluster_silhouette = sample_silhouettes[labels == cluster].mean()
        print(f"   - 클러스터 {cluster}: {cluster_silhouette:.3f}")
    
    return kmeans, labels


def interpret_clusters(df_analysis, feature_cols, labels, original_feature_cols):
    """
    6단계: 클러스터 해석
    - Centroid 분석
    - 클러스터 프로파일링
    """
    print("\n" + "=" * 80)
    print("📖 6단계: 클러스터 해석 및 프로파일링")
    print("=" * 80)
    
    df_result = df_analysis.copy()
    df_result['클러스터'] = labels
    
    # 클러스터별 통계
    print("\n📊 클러스터별 특성 (평균값):")
    
    # 원본 변수 기준 통계
    analysis_cols = [
        '평균_매출', '매출_성장률', '매출_변동성', '매출_추세',
        '점포_수', '총_유동인구_수', '총_상주인구_수', '월_평균_소득_금액',
        '점심_매출_비율', '저녁_매출_비율', '주말_매출_비율'
    ]
    
    cluster_stats = df_result.groupby('클러스터')[analysis_cols].mean()
    
    # 전체 평균 대비 비율로 해석
    overall_mean = df_result[analysis_cols].mean()
    cluster_relative = cluster_stats / overall_mean * 100 - 100  # % 차이
    
    print("\n" + "-" * 80)
    print("클러스터별 평균값 (전체 평균 대비 % 차이)")
    print("-" * 80)
    
    for cluster in sorted(df_result['클러스터'].unique()):
        n_samples = (df_result['클러스터'] == cluster).sum()
        print(f"\n🏷️ 클러스터 {cluster} (n={n_samples})")
        
        for col in analysis_cols:
            val = cluster_stats.loc[cluster, col]
            rel_diff = cluster_relative.loc[cluster, col]
            
            # 단위 포맷팅
            if '매출' in col and '비율' not in col and '성장률' not in col and '변동성' not in col and '추세' not in col:
                val_str = f"{val/1e8:.1f}억원"
            elif '인구' in col:
                val_str = f"{val/1e4:.1f}만명"
            elif '소득' in col:
                val_str = f"{val/1e4:.0f}만원"
            elif '비율' in col or '성장률' in col:
                val_str = f"{val:.1f}%"
            else:
                val_str = f"{val:.1f}"
            
            # 방향 표시
            if rel_diff > 20:
                direction = "▲▲"
            elif rel_diff > 5:
                direction = "▲"
            elif rel_diff < -20:
                direction = "▼▼"
            elif rel_diff < -5:
                direction = "▼"
            else:
                direction = "━"
            
            print(f"   {col}: {val_str} ({direction} {rel_diff:+.1f}%)")
    
    # 클러스터 네이밍
    print("\n" + "=" * 80)
    print("🏷️ 클러스터 프로파일 및 네이밍 제안")
    print("=" * 80)
    
    cluster_profiles = {}
    for cluster in sorted(df_result['클러스터'].unique()):
        growth = cluster_stats.loc[cluster, '매출_성장률']
        sales = cluster_stats.loc[cluster, '평균_매출']
        volatility = cluster_stats.loc[cluster, '매출_변동성']
        trend = cluster_stats.loc[cluster, '매출_추세']
        
        # 특성 기반 분류
        characteristics = []
        
        # 성장성
        if growth > 20:
            characteristics.append("고성장")
        elif growth > 0:
            characteristics.append("안정성장")
        elif growth > -10:
            characteristics.append("정체")
        else:
            characteristics.append("쇠퇴")
        
        # 규모
        overall_sales = df_result['평균_매출'].median()
        if sales > overall_sales * 2:
            characteristics.append("대형")
        elif sales > overall_sales:
            characteristics.append("중형")
        else:
            characteristics.append("소형")
        
        # 안정성
        if volatility < 30:
            characteristics.append("안정")
        elif volatility > 50:
            characteristics.append("변동")
        
        profile_name = " ".join(characteristics) + " 상권"
        cluster_profiles[cluster] = profile_name
        
        print(f"\n클러스터 {cluster}: 「{profile_name}」")
        print(f"   - 성장률: {growth:.1f}%")
        print(f"   - 평균매출: {sales/1e8:.1f}억원")
        print(f"   - 변동성: {volatility:.1f}%")
        print(f"   - 추세: {trend:.2f}")
    
    df_result['클러스터_명'] = df_result['클러스터'].map(cluster_profiles)
    
    return df_result, cluster_stats, cluster_profiles


def visualize_clusters(df_result, X_pca, labels, cluster_profiles):
    """
    7단계: 클러스터 시각화
    """
    print("\n" + "=" * 80)
    print("📊 7단계: 클러스터 시각화")
    print("=" * 80)
    
    n_clusters = len(cluster_profiles)
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    
    # 1. PCA 2D 산점도
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for cluster in sorted(cluster_profiles.keys()):
        mask = labels == cluster
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   label=f'{cluster}: {cluster_profiles[cluster]}',
                   alpha=0.6, s=50, color=colors[cluster])
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('상권 클러스터링 결과 (PCA 2D 투영)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_cluster_pca_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 클러스터별 주요 지표 박스플롯
    key_features = ['평균_매출', '매출_성장률', '매출_변동성', '총_유동인구_수']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(key_features):
        data_to_plot = []
        cluster_labels = []
        
        for cluster in sorted(cluster_profiles.keys()):
            data = df_result[df_result['클러스터'] == cluster][feature].values
            data_to_plot.append(data)
            cluster_labels.append(f'{cluster}: {cluster_profiles[cluster][:6]}')
        
        bp = axes[i].boxplot(data_to_plot, labels=cluster_labels, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        axes[i].set_title(feature)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_cluster_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. 클러스터별 레이더 차트
    categories = ['매출_성장률', '매출_변동성', '점포_수', '총_유동인구_수', '월_평균_소득_금액', '주말_매출_비율']
    
    # 정규화 (0-1)
    normalized_stats = df_result.groupby('클러스터')[categories].mean()
    for col in categories:
        min_val = normalized_stats[col].min()
        max_val = normalized_stats[col].max()
        if max_val > min_val:
            normalized_stats[col] = (normalized_stats[col] - min_val) / (max_val - min_val)
        else:
            normalized_stats[col] = 0.5
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    for cluster in sorted(cluster_profiles.keys()):
        values = normalized_stats.loc[cluster].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=f'{cluster}: {cluster_profiles[cluster]}', color=colors[cluster])
        ax.fill(angles, values, alpha=0.15, color=colors[cluster])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title('클러스터별 특성 프로파일 (레이더 차트)')
    ax.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_cluster_radar.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. 상권 유형(골목/발달/전통시장) vs 클러스터 교차표
    if '상권_구분_코드_명' in df_result.columns:
        crosstab = pd.crosstab(df_result['상권_구분_코드_명'], df_result['클러스터_명'], normalize='index') * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        crosstab.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
        ax.set_xlabel('기존 상권 유형')
        ax.set_ylabel('비율 (%)')
        ax.set_title('기존 상권 유형별 클러스터 분포')
        ax.legend(title='클러스터', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/06_cluster_vs_area_type.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print("✅ 시각화 완료")
    print(f"   - 저장 위치: {OUTPUT_DIR}/")


def save_results(df_result, cluster_stats, cluster_profiles):
    """
    8단계: 결과 저장
    """
    print("\n" + "=" * 80)
    print("💾 8단계: 결과 저장")
    print("=" * 80)
    
    # 1. 상권별 클러스터 결과
    output_cols = ['상권_코드', '상권_코드_명', '상권_구분_코드_명', '클러스터', '클러스터_명',
                   '평균_매출', '매출_성장률', '매출_변동성', '매출_추세',
                   '점포_수', '총_유동인구_수', '월_평균_소득_금액']
    
    df_result[output_cols].to_csv(f"{OUTPUT_DIR}/clustering_result.csv", index=False, encoding='utf-8-sig')
    
    # 2. 클러스터 통계
    cluster_stats.to_csv(f"{OUTPUT_DIR}/cluster_statistics.csv", encoding='utf-8-sig')
    
    # 3. 클러스터 프로파일 요약
    profile_df = pd.DataFrame([
        {'클러스터': k, '프로파일': v, '상권수': (df_result['클러스터'] == k).sum()}
        for k, v in cluster_profiles.items()
    ])
    profile_df.to_csv(f"{OUTPUT_DIR}/cluster_profiles.csv", index=False, encoding='utf-8-sig')
    
    print("✅ 결과 저장 완료")
    print(f"   - {OUTPUT_DIR}/clustering_result.csv")
    print(f"   - {OUTPUT_DIR}/cluster_statistics.csv")
    print(f"   - {OUTPUT_DIR}/cluster_profiles.csv")


def generate_business_insights(df_result, cluster_profiles, cluster_stats):
    """
    9단계: 비즈니스 인사이트 도출
    """
    print("\n" + "=" * 80)
    print("💡 9단계: 비즈니스 인사이트")
    print("=" * 80)
    
    print("\n📌 주요 발견 사항:")
    
    # 성장률 기준 분류
    growth_clusters = cluster_stats['매출_성장률'].sort_values(ascending=False)
    best_growth = growth_clusters.index[0]
    worst_growth = growth_clusters.index[-1]
    
    print(f"\n1️⃣ 가장 성장한 상권 유형: 클러스터 {best_growth} ({cluster_profiles[best_growth]})")
    print(f"   - 매출 성장률: {cluster_stats.loc[best_growth, '매출_성장률']:.1f}%")
    print(f"   - 상권 수: {(df_result['클러스터'] == best_growth).sum()}개")
    
    print(f"\n2️⃣ 가장 쇠퇴한 상권 유형: 클러스터 {worst_growth} ({cluster_profiles[worst_growth]})")
    print(f"   - 매출 성장률: {cluster_stats.loc[worst_growth, '매출_성장률']:.1f}%")
    print(f"   - 상권 수: {(df_result['클러스터'] == worst_growth).sum()}개")
    
    # 가장 큰 클러스터
    largest_cluster = df_result['클러스터'].value_counts().idxmax()
    print(f"\n3️⃣ 가장 많은 상권 유형: 클러스터 {largest_cluster} ({cluster_profiles[largest_cluster]})")
    print(f"   - 상권 수: {(df_result['클러스터'] == largest_cluster).sum()}개")
    
    print("\n📌 의사결정 시사점:")
    print("\n[창업자 관점]")
    print(f"   • 성장 잠재력이 높은 「{cluster_profiles[best_growth]}」 유형 상권 우선 검토")
    print(f"   • 「{cluster_profiles[worst_growth]}」 유형은 신중한 진입 필요")
    
    print("\n[정책 관점]")
    print(f"   • 「{cluster_profiles[worst_growth]}」 유형 상권에 대한 활성화 정책 필요")
    print("   • 클러스터별 맞춤형 지원 전략 수립 가능")
    
    # 결과 파일로 저장
    with open(f"{OUTPUT_DIR}/business_insights.txt", 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("서울시 한식음식점 상권 유형화 분석 - 비즈니스 인사이트\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("📌 클러스터 프로파일 요약\n")
        f.write("-" * 40 + "\n")
        for cluster, profile in cluster_profiles.items():
            n = (df_result['클러스터'] == cluster).sum()
            growth = cluster_stats.loc[cluster, '매출_성장률']
            sales = cluster_stats.loc[cluster, '평균_매출'] / 1e8
            f.write(f"클러스터 {cluster}: {profile}\n")
            f.write(f"  - 상권 수: {n}개\n")
            f.write(f"  - 평균 매출: {sales:.1f}억원\n")
            f.write(f"  - 성장률: {growth:.1f}%\n\n")
        
        f.write("\n📌 전략적 시사점\n")
        f.write("-" * 40 + "\n")
        f.write(f"• 성장 추천: {cluster_profiles[best_growth]}\n")
        f.write(f"• 주의 필요: {cluster_profiles[worst_growth]}\n")
    
    print(f"\n✅ 인사이트 저장: {OUTPUT_DIR}/business_insights.txt")


def main():
    """
    메인 실행 함수
    """
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  📊 서울시 한식음식점 상권 유형화 분석 (K-means Clustering)".center(76) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    
    # 1. 데이터 로드
    df = load_data()
    
    # 2. 시계열 파생변수 생성
    features_df = extract_time_features(df)
    
    # 3. 전처리
    df_analysis, feature_cols = preprocess_features(features_df)
    
    # 4. 표준화 및 PCA
    X_scaled, X_pca, pca, scaler, loadings = standardize_and_pca(df_analysis, feature_cols)
    
    # 5-1. 최적 k 탐색
    optimal_k, silhouettes = find_optimal_k(X_pca, max_k=8)
    
    # 5-2. K-means 클러스터링
    kmeans, labels = perform_kmeans(X_pca, optimal_k)
    
    # 6. 클러스터 해석
    df_result, cluster_stats, cluster_profiles = interpret_clusters(
        df_analysis, feature_cols, labels, 
        ['평균_매출', '매출_성장률', '매출_변동성', '점포_수', '총_유동인구_수']
    )
    
    # 7. 시각화
    visualize_clusters(df_result, X_pca, labels, cluster_profiles)
    
    # 8. 결과 저장
    save_results(df_result, cluster_stats, cluster_profiles)
    
    # 9. 비즈니스 인사이트
    generate_business_insights(df_result, cluster_profiles, cluster_stats)
    
    print("\n")
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  ✅ 분석 완료!".center(76) + "█")
    print("█" + f"  결과 저장 위치: {OUTPUT_DIR}/".center(76) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)
    print("\n")


if __name__ == "__main__":
    main()

