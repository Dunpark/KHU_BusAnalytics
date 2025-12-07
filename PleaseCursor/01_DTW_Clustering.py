# -*- coding: utf-8 -*-
"""
01_DTW_Clustering.py
DTW 기반 시계열 클러스터링
- 선행연구 방법론 적용: Dynamic Time Warping 기반 K-means 클러스터링
- 점포당 매출액 시계열 패턴으로 상권 유형 분류
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# DTW 클러스터링을 위한 라이브러리
try:
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance
    TSLEARN_AVAILABLE = True
except ImportError:
    TSLEARN_AVAILABLE = False
    print("Warning: tslearn이 설치되지 않았습니다. pip install tslearn 실행 필요")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("DTW 기반 시계열 클러스터링")
print("=" * 60)

# 데이터 로드
df_ts = pd.read_csv('df_cafe_timeSeries.csv')
print(f"\n데이터 로드 완료: {df_ts.shape}")

# ============================================================
# 1. 점포당 매출액 시계열 데이터 준비
# ============================================================
print("\n" + "=" * 60)
print("1. 점포당 매출액 시계열 데이터 준비")
print("=" * 60)

# 점포당 매출액 계산
df_ts['점포당_매출액'] = df_ts['당월_매출_금액'] / df_ts['유사_업종_점포_수']
df_ts['점포당_매출액'] = df_ts['점포당_매출액'].replace([np.inf, -np.inf], np.nan)

# 피벗 테이블 생성 (상권 x 분기)
pivot_sps = df_ts.pivot_table(
    index='상권_코드',
    columns='기준_년분기_코드',
    values='점포당_매출액',
    aggfunc='mean'
)

print(f"피벗 테이블 shape: {pivot_sps.shape}")
print(f"분기: {list(pivot_sps.columns)}")

# 결측치 처리 (16개 분기 모두 있는 상권만 선택)
pivot_sps_clean = pivot_sps.dropna()
print(f"결측치 제거 후: {pivot_sps_clean.shape}")

# 시계열 데이터 준비 (3D array: n_samples, n_timestamps, n_features)
ts_data = pivot_sps_clean.values
n_samples, n_timestamps = ts_data.shape
ts_data_3d = ts_data.reshape(n_samples, n_timestamps, 1)

print(f"시계열 데이터 shape: {ts_data_3d.shape}")

# ============================================================
# 2. DTW K-means 클러스터링
# ============================================================
print("\n" + "=" * 60)
print("2. DTW K-means 클러스터링")
print("=" * 60)

if TSLEARN_AVAILABLE:
    # 데이터 스케일링
    scaler = TimeSeriesScalerMeanVariance()
    ts_data_scaled = scaler.fit_transform(ts_data_3d)
    
    # 최적의 K 탐색 (Elbow Method + Silhouette Score)
    print("\n최적 K 탐색 중...")
    K_range = range(2, 8)
    inertias = []
    silhouettes = []
    
    for k in K_range:
        print(f"  K={k} 테스트 중...")
        model = TimeSeriesKMeans(n_clusters=k, metric="dtw", max_iter=10, 
                                  random_state=42, n_init=2, verbose=0)
        labels = model.fit_predict(ts_data_scaled)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(ts_data_scaled.reshape(n_samples, -1), labels))
    
    # 시각화: Elbow + Silhouette
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax1.set_ylabel('Inertia', fontsize=12)
    ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(K_range, silhouettes, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Score', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dtw_00_optimal_k.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[저장] dtw_00_optimal_k.png")
    
    # 선행연구 기반 K=5 적용 (5가지 상권 유형)
    n_clusters = 5
    print(f"\nDTW K-means 클러스터링 (K={n_clusters}) 실행 중...")
    
    model_dtw = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", 
                                  max_iter=15, random_state=42, n_init=3, verbose=0)
    clusters = model_dtw.fit_predict(ts_data_scaled)
    
    # 클러스터 결과 저장
    df_clusters = pd.DataFrame({
        '상권_코드': pivot_sps_clean.index,
        'DTW_Cluster': clusters
    })
    
    print(f"\n클러스터 분포:")
    print(df_clusters['DTW_Cluster'].value_counts().sort_index())
    
    # ============================================================
    # 3. 클러스터별 특성 분석 및 상권유형 매핑
    # ============================================================
    print("\n" + "=" * 60)
    print("3. 클러스터별 특성 분석")
    print("=" * 60)
    
    # 각 클러스터의 평균 시계열 추출
    cluster_centers = model_dtw.cluster_centers_
    
    # 클러스터별 특성 계산
    cluster_stats = []
    for c in range(n_clusters):
        cluster_mask = clusters == c
        cluster_data = ts_data[cluster_mask]
        
        # 평균 점포당 매출액
        mean_sps = np.mean(cluster_data)
        
        # 성장률 (첫 분기 대비 마지막 분기)
        first_vals = cluster_data[:, 0]
        last_vals = cluster_data[:, -1]
        growth_rate = np.mean((last_vals - first_vals) / first_vals) * 100
        
        # 변동계수
        cv = np.mean(np.std(cluster_data, axis=1) / np.mean(cluster_data, axis=1))
        
        cluster_stats.append({
            'Cluster': c,
            'Count': np.sum(cluster_mask),
            'Mean_SPS': mean_sps / 1e6,  # 백만원 단위
            'Growth_Rate': growth_rate,
            'CV': cv
        })
    
    df_cluster_stats = pd.DataFrame(cluster_stats)
    print("\n클러스터별 특성:")
    print(df_cluster_stats.to_string(index=False))
    
    # 상권유형 매핑 (클러스터 특성 기반)
    # 성장률과 규모를 기준으로 5가지 유형 매핑
    def map_cluster_to_type(row):
        """
        클러스터를 상권유형으로 매핑
        - 성장상권: 높은 성장률 + 큰 규모
        - 기대상권: 높은 성장률 + 작은/중간 규모
        - 전통강호상권: 낮은 성장률 + 큰 규모
        - 정체상권: 낮은 성장률 + 중간 규모
        - 쇠퇴상권: 음의 성장률
        """
        growth = row['Growth_Rate']
        scale = row['Mean_SPS']
        
        # 성장률 기준 (전체 평균 대비)
        avg_growth = df_cluster_stats['Growth_Rate'].mean()
        avg_scale = df_cluster_stats['Mean_SPS'].mean()
        
        if growth < -5:  # 음의 성장률
            return '쇠퇴상권'
        elif growth > avg_growth:  # 평균 이상 성장
            if scale > avg_scale:
                return '성장상권'
            else:
                return '기대상권'
        else:  # 평균 이하 성장
            if scale > avg_scale * 1.2:
                return '전통강호상권'
            else:
                return '정체상권'
    
    df_cluster_stats['상권유형'] = df_cluster_stats.apply(map_cluster_to_type, axis=1)
    
    # 클러스터-유형 매핑 딕셔너리
    cluster_to_type = dict(zip(df_cluster_stats['Cluster'], df_cluster_stats['상권유형']))
    df_clusters['상권유형'] = df_clusters['DTW_Cluster'].map(cluster_to_type)
    
    print("\n클러스터-상권유형 매핑:")
    print(df_cluster_stats[['Cluster', '상권유형', 'Count', 'Mean_SPS', 'Growth_Rate']].to_string(index=False))
    
    # ============================================================
    # 4. 시각화
    # ============================================================
    print("\n" + "=" * 60)
    print("4. 클러스터링 결과 시각화")
    print("=" * 60)
    
    # 4-1. 클러스터별 시계열 패턴
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    type_names = ['성장상권', '기대상권', '전통강호상권', '정체상권', '쇠퇴상권']
    
    for c in range(n_clusters):
        ax = axes[c]
        cluster_mask = clusters == c
        cluster_ts = ts_data_scaled[cluster_mask]
        
        # 샘플 시계열 플롯 (최대 20개)
        for i, ts in enumerate(cluster_ts[:20]):
            ax.plot(ts.flatten(), alpha=0.3, color=colors[c % len(colors)])
        
        # 클러스터 중심
        ax.plot(cluster_centers[c].flatten(), color='black', linewidth=3, 
                label='Cluster Center')
        
        area_type = cluster_to_type.get(c, f'Cluster {c}')
        ax.set_title(f'{area_type}\n(n={np.sum(cluster_mask)})', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Quarter')
        ax.set_ylabel('Normalized SPS')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    # 마지막 subplot: 클러스터 중심 비교
    ax = axes[5]
    for c in range(n_clusters):
        area_type = cluster_to_type.get(c, f'Cluster {c}')
        ax.plot(cluster_centers[c].flatten(), linewidth=2, 
                label=area_type, color=colors[c % len(colors)])
    ax.set_title('클러스터 중심 비교', fontsize=12, fontweight='bold')
    ax.set_xlabel('Quarter')
    ax.set_ylabel('Normalized SPS')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dtw_01_clustering_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[저장] dtw_01_clustering_results.png")
    
    # 4-2. 상권유형별 특성 비교
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 상권유형별 분포
    ax1 = axes[0, 0]
    type_counts = df_clusters['상권유형'].value_counts()
    bars = ax1.bar(type_counts.index, type_counts.values, color=colors[:len(type_counts)], alpha=0.8)
    ax1.set_title('상권유형별 분포', fontsize=12, fontweight='bold')
    ax1.set_ylabel('상권 수')
    ax1.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, type_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(val), ha='center', va='bottom')
    
    # 성장률 분포
    ax2 = axes[0, 1]
    growth_by_type = df_cluster_stats.set_index('상권유형')['Growth_Rate']
    bars = ax2.bar(growth_by_type.index, growth_by_type.values, 
                   color=[colors[i % len(colors)] for i in range(len(growth_by_type))], alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('상권유형별 평균 성장률', fontsize=12, fontweight='bold')
    ax2.set_ylabel('성장률 (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    # 평균 매출 분포
    ax3 = axes[1, 0]
    sps_by_type = df_cluster_stats.set_index('상권유형')['Mean_SPS']
    bars = ax3.bar(sps_by_type.index, sps_by_type.values,
                   color=[colors[i % len(colors)] for i in range(len(sps_by_type))], alpha=0.8)
    ax3.set_title('상권유형별 평균 점포당 매출액', fontsize=12, fontweight='bold')
    ax3.set_ylabel('점포당 매출액 (백만원)')
    ax3.tick_params(axis='x', rotation=45)
    
    # 변동계수 분포
    ax4 = axes[1, 1]
    cv_by_type = df_cluster_stats.set_index('상권유형')['CV']
    bars = ax4.bar(cv_by_type.index, cv_by_type.values,
                   color=[colors[i % len(colors)] for i in range(len(cv_by_type))], alpha=0.8)
    ax4.set_title('상권유형별 변동계수 (안정성)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('변동계수')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('dtw_02_type_characteristics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[저장] dtw_02_type_characteristics.png")
    
    # ============================================================
    # 5. 결과 저장
    # ============================================================
    print("\n" + "=" * 60)
    print("5. 결과 저장")
    print("=" * 60)
    
    # 기존 특성 데이터와 병합
    df_ts_features = pd.read_csv('df_ts_features.csv')
    
    # DTW 클러스터 결과 병합
    df_merged = df_ts_features.merge(
        df_clusters[['상권_코드', 'DTW_Cluster', '상권유형']],
        on='상권_코드',
        how='left',
        suffixes=('_rule', '_dtw')
    )
    
    # DTW 기반 상권유형으로 업데이트 (있는 경우)
    df_merged['상권유형'] = df_merged['상권유형_dtw'].fillna(df_merged['상권유형_rule'])
    df_merged = df_merged.drop(columns=['상권유형_rule', '상권유형_dtw'], errors='ignore')
    
    # DTW 클러스터 결과만 별도 저장
    df_clusters_final = df_clusters.merge(
        df_ts_features[['상권_코드', 'sps_mean', 'sps_growth', 'sps_cv', 'sps_first', 'sps_last']],
        on='상권_코드',
        how='left'
    )
    df_clusters_final.to_csv('df_dtw_clusters.csv', index=False, encoding='utf-8-sig')
    print("\n[저장] df_dtw_clusters.csv")
    
    # 상권유형별 요약 통계
    type_summary = df_clusters_final.groupby('상권유형').agg({
        'sps_mean': ['mean', 'std', 'count'],
        'sps_growth': ['mean', 'std'],
        'sps_cv': ['mean', 'std']
    }).round(4)
    type_summary.to_csv('df_dtw_type_summary.csv', encoding='utf-8-sig')
    print("[저장] df_dtw_type_summary.csv")
    
    print("\n" + "=" * 60)
    print("DTW 클러스터링 완료!")
    print("=" * 60)

else:
    print("\ntslearn 라이브러리가 필요합니다.")
    print("설치: pip install tslearn")
    print("\n규칙 기반 분류 결과를 대신 사용합니다.")
    
    # 규칙 기반 결과 로드
    df_ts_features = pd.read_csv('df_ts_features.csv')
    
    # DTW_Cluster 없이 저장
    df_clusters = df_ts_features[['상권_코드', '상권유형']].copy()
    df_clusters['DTW_Cluster'] = -1  # DTW 미적용 표시
    df_clusters.to_csv('df_dtw_clusters.csv', index=False, encoding='utf-8-sig')
    print("\n[저장] df_dtw_clusters.csv (규칙 기반)")

