"""
download_dataset.py
-------------------
Roboflow에서 Rock-Scissors-Paper 데이터셋을 자동으로 다운로드하고
YOLO 포맷으로 data/ 폴더에 배치합니다.

사용법:
    pip install roboflow
    python download_dataset.py --api-key YOUR_ROBOFLOW_API_KEY

Roboflow API 키 없이 수동으로 데이터를 준비하려면
아래 "수동 데이터 준비 가이드"를 참고하세요.

수동 데이터 준비 가이드:
    1. https://public.roboflow.com/object-detection/rock-paper-scissors-sxsw 에서
       "YOLO v8" 포맷으로 다운로드
    2. 압축 해제 후 아래 구조로 복사:
         data/train/images/  <- 학습 이미지 (.jpg/.png)
         data/train/labels/  <- 학습 레이블 (.txt, YOLO 포맷)
         data/valid/images/  <- 검증 이미지
         data/valid/labels/  <- 검증 레이블
         data/test/images/   <- 테스트 이미지 (선택)
         data/test/labels/   <- 테스트 레이블 (선택)
    3. 레이블 파일 포맷 (각 줄): <class_id> <cx> <cy> <w> <h>
         0 = rock, 1 = scissors, 2 = paper
         모든 값은 0~1 사이로 정규화된 값

웹캠으로 직접 데이터를 수집하려면 collect_data.py를 실행하세요.
"""

import argparse
import os
import sys
import shutil
from pathlib import Path


def download_from_roboflow(api_key: str, workspace: str, project: str, version: int):
    """Roboflow에서 데이터셋을 다운로드하고 data/ 폴더에 배치합니다."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] roboflow 패키지가 없습니다. pip install roboflow 실행 후 재시도하세요.")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    dataset = project_obj.version(version).download("yolov8")

    dataset_dir = Path(dataset.location)
    dest_root = Path("data")

    # train / valid / test 폴더 이동
    for split in ["train", "valid", "test"]:
        src_images = dataset_dir / split / "images"
        src_labels = dataset_dir / split / "labels"
        dst_images = dest_root / split / "images"
        dst_labels = dest_root / split / "labels"

        if src_images.exists():
            dst_images.mkdir(parents=True, exist_ok=True)
            for f in src_images.iterdir():
                shutil.copy2(f, dst_images / f.name)
            print(f"  [OK] {split}/images: {len(list(src_images.iterdir()))}장 복사")

        if src_labels.exists():
            dst_labels.mkdir(parents=True, exist_ok=True)
            for f in src_labels.iterdir():
                shutil.copy2(f, dst_labels / f.name)
            print(f"  [OK] {split}/labels: {len(list(src_labels.iterdir()))}개 복사")

    # 임시 다운로드 폴더 삭제
    shutil.rmtree(dataset_dir, ignore_errors=True)
    print("\n[완료] 데이터셋 준비 완료!")


def check_dataset():
    """현재 data/ 폴더의 상태를 출력합니다."""
    root = Path("data")
    for split in ["train", "valid", "test"]:
        images = list((root / split / "images").glob("*.*")) if (root / split / "images").exists() else []
        labels = list((root / split / "labels").glob("*.txt")) if (root / split / "labels").exists() else []
        print(f"  {split:6s} : images={len(images):5d}  labels={len(labels):5d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rock-Scissors-Paper 데이터셋 다운로드")
    parser.add_argument("--api-key",   type=str, default="", help="Roboflow API 키")
    parser.add_argument("--workspace", type=str, default="joseph-nelson",
                        help="Roboflow 워크스페이스 이름")
    parser.add_argument("--project",   type=str, default="rock-paper-scissors-sxsw",
                        help="Roboflow 프로젝트 이름")
    parser.add_argument("--version",   type=int, default=14,
                        help="데이터셋 버전")
    parser.add_argument("--check",     action="store_true",
                        help="현재 데이터셋 상태만 확인")
    args = parser.parse_args()

    print("=== 데이터셋 현황 ===")
    check_dataset()

    if args.check:
        sys.exit(0)

    if not args.api_key:
        print("\n[INFO] --api-key 가 제공되지 않았습니다.")
        print("       수동으로 data/ 폴더에 이미지/레이블을 배치하거나")
        print("       --api-key YOUR_KEY 옵션을 추가하여 재실행하세요.")
        sys.exit(0)

    print("\n=== Roboflow에서 데이터셋 다운로드 중... ===")
    download_from_roboflow(args.api_key, args.workspace, args.project, args.version)

    print("\n=== 다운로드 후 데이터셋 현황 ===")
    check_dataset()
