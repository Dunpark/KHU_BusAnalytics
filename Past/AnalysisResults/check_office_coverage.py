"""
기준_년분기_코드 + 상권_코드 조합이
직장인구 데이터에 모두 존재하는지 확인
"""

import pandas as pd
import sys

MERGED_PATH = "../Merged_datasets/4개년_통합데이터_추정매출_상주인구_소득소비_길단위인구_점포_영역.csv"
OFFICE_PATH = "../Data_Raw_new/서울시 상권분석서비스(직장인구-상권).csv"


def read_with_encodings(path, encodings=("utf-8", "utf-8-sig", "cp949", "euc-kr")):
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise RuntimeError(f"Cannot read file with tried encodings: {encodings}")


def main():
    print("=" * 80)
    print("📊 기준_년분기_코드 × 상권_코드 조합 매핑 검사")
    print("=" * 80)

    merged = pd.read_csv(
        MERGED_PATH,
        encoding="utf-8",
        usecols=["기준_년분기_코드", "상권_코드"],
    )
    merged["기준_년분기_코드"] = merged["기준_년분기_코드"].astype(int)
    merged["상권_코드"] = merged["상권_코드"].astype(int)

    office = read_with_encodings(OFFICE_PATH)
    # 직장인구 파일은 보통 열 이름이 같거나 대소문자/언더스코어 차이 없음
    if "기준_년분기_코드" not in office.columns or "상권_코드" not in office.columns:
        print(f"❌ 직장인구 파일에 필요한 열이 없습니다. 열 목록: {office.columns.tolist()}")
        sys.exit(1)

    office = office[["기준_년분기_코드", "상권_코드"]].copy()
    office["기준_년분기_코드"] = office["기준_년분기_코드"].astype(int)
    office["상권_코드"] = office["상권_코드"].astype(int)

    merged_combos = (
        merged.drop_duplicates(subset=["기준_년분기_코드", "상권_코드"])
        .assign(key=lambda x: list(zip(x["기준_년분기_코드"], x["상권_코드"])))
        ["key"]
    )
    office_combos = (
        office.drop_duplicates(subset=["기준_년분기_코드", "상권_코드"])
        .assign(key=lambda x: list(zip(x["기준_년분기_코드"], x["상권_코드"])))
        ["key"]
    )

    merged_set = set(merged_combos)
    office_set = set(office_combos)

    missing = merged_set - office_set

    total = len(merged_set)
    missing_cnt = len(missing)
    covered = total - missing_cnt
    missing_ratio = missing_cnt / total * 100 if total else 0

    print(f"\n[조합 개수]")
    print(f"  • Merged 데이터 조합 수: {total:,}")
    print(f"  • 직장인구 데이터 조합 수: {len(office_set):,}")

    print(f"\n[매핑 결과]")
    print(f"  • 매핑됨: {covered:,}개 ({100 - missing_ratio:.2f}%)")
    print(f"  • 매핑 안 됨: {missing_cnt:,}개 ({missing_ratio:.2f}%)")

    # 누락된 조합 샘플 20개 출력
    if missing_cnt > 0:
        print("\n[매핑 안 된 조합 샘플 20개] (기준_년분기_코드, 상권_코드)")
        for tup in list(missing)[:20]:
            print(f"  {tup[0]}, {tup[1]}")

    print("\n✅ 완료")


if __name__ == "__main__":
    main()


