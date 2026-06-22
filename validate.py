"""
validate.py
-----------
학습된 모델을 검증 데이터셋으로 평가하고 성능 지표를 출력합니다.

사용법:
    python validate.py --model runs/detect/rps_finetune/weights/best.pt
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Rock-Scissors-Paper 모델 검증")
    parser.add_argument("--model", type=str, required=True,
                        help="평가할 모델 가중치 경로 (best.pt)")
    parser.add_argument("--data",  type=str, default="data/rps.yaml",
                        help="데이터셋 yaml 파일")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="입력 이미지 크기")
    parser.add_argument("--conf",  type=float, default=0.001,
                        help="신뢰도 임계값 (mAP 계산 시 낮게 설정)")
    parser.add_argument("--iou",   type=float, default=0.6,
                        help="NMS IoU 임계값")
    parser.add_argument("--device",type=str,  default="",
                        help="장치 ('0', 'cpu')")
    parser.add_argument("--split", type=str,  default="val",
                        choices=["train", "val", "test"],
                        help="평가할 데이터 분할")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] pip install ultralytics 후 재시도하세요.")
        raise

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일 없음: {model_path}")

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(Path(args.data).resolve()),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device if args.device else None,
        split=args.split,
        plots=True,
        verbose=True,
    )

    print(f"\n{'='*50}")
    print(f"  평가 결과 ({args.split})")
    print(f"{'='*50}")
    print(f"  mAP@50        : {metrics.box.map50:.4f}")
    print(f"  mAP@50:95     : {metrics.box.map:.4f}")
    print(f"  Precision     : {metrics.box.mp:.4f}")
    print(f"  Recall        : {metrics.box.mr:.4f}")

    names = model.names
    print(f"\n  클래스별 AP@50:")
    for i, ap in enumerate(metrics.box.ap50):
        cls_name = names.get(i, str(i))
        print(f"    {cls_name:12s}: {ap:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
