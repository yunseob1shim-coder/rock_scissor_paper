"""
predict.py
----------
학습된 Rock-Scissors-Paper 모델로 추론을 실행합니다.

사용법:
    python predict.py --model runs/detect/rps_finetune/weights/best.pt --source 0
    python predict.py --model runs/detect/rps_finetune/weights/best.pt --source image.jpg
    python predict.py --model runs/detect/rps_finetune/weights/best.pt --source video.mp4
    python predict.py --model runs/detect/rps_finetune/weights/best.pt --source data/test/images/
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Rock-Scissors-Paper 추론")
    parser.add_argument("--model",  type=str,   required=True,
                        help="모델 가중치 경로 (best.pt)")
    parser.add_argument("--source", type=str,   default="0",
                        help="입력 소스: 웹캠 번호(0), 이미지, 영상, 폴더 경로")
    parser.add_argument("--imgsz",  type=int,   default=640,
                        help="추론 이미지 크기")
    parser.add_argument("--conf",   type=float, default=0.25,
                        help="최소 신뢰도 임계값")
    parser.add_argument("--iou",    type=float, default=0.45,
                        help="NMS IoU 임계값")
    parser.add_argument("--device", type=str,   default="",
                        help="장치 ('0', 'cpu')")
    parser.add_argument("--save",   action="store_true", default=True,
                        help="결과 이미지/영상 저장")
    parser.add_argument("--show",   action="store_true", default=False,
                        help="실시간 화면 출력 (웹캠/영상에서 권장)")
    parser.add_argument("--project",type=str,   default="runs/predict",
                        help="결과 저장 디렉토리")
    parser.add_argument("--name",   type=str,   default="rps_result",
                        help="실험 이름")
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

    # source가 숫자 문자열이면 정수(웹캠 인덱스)로 변환
    source = int(args.source) if args.source.isdigit() else args.source

    results = model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device if args.device else None,
        save=args.save,
        show=args.show,
        project=args.project,
        name=args.name,
        verbose=True,
    )

    # 결과 요약
    total = sum(len(r.boxes) for r in results)
    print(f"\n검출된 객체 수 (전체): {total}")
    save_dir = Path(args.project) / args.name
    print(f"결과 저장 위치: {save_dir}")


if __name__ == "__main__":
    main()
