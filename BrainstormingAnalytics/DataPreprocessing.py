"""
서비스_업종_코드_명이 '한식음식점'인 행만 필터링하여 별도 CSV로 저장하는 스크립트.

실행 방법: 프로젝트 루트(KHU_BusAnalytics)에서
    python BrainstormingAnalytics/DataPreprocessing.py
"""

import os
import pandas as pd


def main():
    # 입력/출력 경로 설정
    src = "Merged_datasets/4개년_통합데이터_추정매출_상주인구_소득소비_길단위인구_점포_영역.csv"
    dst_dir = "BrainstormingAnalytics"
    dst_name = "4개년_한식음식점_통합데이터_추정매출_상주인구_소득소비_길단위인구_점포_영역.csv"
    dst_path = os.path.join(dst_dir, dst_name)

    # 출력 폴더 생성
    os.makedirs(dst_dir, exist_ok=True)

    # 필터링 대상 컬럼 및 값
    filter_col = "서비스_업종_코드_명"
    filter_value = "한식음식점"

    # 전체 데이터 로드
    print("데이터 로딩 중...")
    df = pd.read_csv(src, encoding="utf-8")
    print(f"원본 행 수: {len(df):,}")

    # 컬럼 존재 확인
    if filter_col not in df.columns:
        raise ValueError(f"'{filter_col}' 컬럼이 존재하지 않습니다.")

    # 한식음식점만 필터링
    filtered = df[df[filter_col] == filter_value].copy()
    print(f"필터링 컬럼: {filter_col}")
    print(f"필터링 값: {filter_value}")
    print(f"필터링 후 행 수: {len(filtered):,}")

    # 저장
    filtered.to_csv(dst_path, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {dst_path}")


if __name__ == "__main__":
    main()

