"""
train.py
--------
yolo26n.pt 모델을 가위바위보(Rock-Scissors-Paper) 탐지용으로 파인튜닝합니다.

아키텍처:  Ultralytics YOLO (C3k2 + C2PSA + SPPF + Detect 헤드)
사전학습:  COCO 80-class
Fine-tune: 3-class (rock / scissors / paper)

사용법:
    python train.py                        # 기본 설정으로 학습
    python train.py --epochs 100 --batch 16
    python train.py --resume                # 중단된 학습 재개
    python train.py --device cpu           # CPU로 학습

출력:
    runs/detect/rps_finetune/              # 학습 결과 디렉토리
        weights/best.pt                    # 최고 성능 가중치
        weights/last.pt                    # 마지막 체크포인트
        results.csv                        # 학습 지표 기록
        confusion_matrix.png               # 혼동 행렬
"""

import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Rock-Scissors-Paper YOLO Fine-tuning")

    # ── 데이터 관련 ──────────────────────────────────────────────
    parser.add_argument("--data",    type=str,   default="data/rps.yaml",
                        help="데이터셋 yaml 파일 경로")
    parser.add_argument("--model",   type=str,   default="yolo26n.pt",
                        help="사전학습 가중치 경로")

    # ── 학습 하이퍼파라미터 ──────────────────────────────────────
    parser.add_argument("--epochs",  type=int,   default=50,
                        help="총 학습 에폭 수 (기본 50)")
    parser.add_argument("--batch",   type=int,   default=16,
                        help="배치 크기; GPU 메모리에 맞게 조정")
    parser.add_argument("--imgsz",   type=int,   default=640,
                        help="입력 이미지 크기 (픽셀)")
    parser.add_argument("--lr0",     type=float, default=0.01,
                        help="초기 학습률")
    parser.add_argument("--lrf",     type=float, default=0.001,
                        help="최종 학습률 (lr0 * lrf)")
    parser.add_argument("--momentum",type=float, default=0.937,
                        help="SGD 모멘텀")
    parser.add_argument("--weight-decay", type=float, default=0.0005,
                        help="가중치 감쇠 (L2 정규화)")
    parser.add_argument("--warmup-epochs",type=float,default=3.0,
                        help="학습률 워밍업 에폭 수")

    # ── Freeze (레이어 동결) ─────────────────────────────────────
    parser.add_argument("--freeze",  type=int,   default=10,
                        help="동결할 레이어 수 (백본 일부 고정). "
                             "0이면 전체 파인튜닝, 높을수록 백본 더 많이 고정")

    # ── 증강 ────────────────────────────────────────────────────
    parser.add_argument("--hsv-h",   type=float, default=0.015,
                        help="Hue 증강 범위")
    parser.add_argument("--hsv-s",   type=float, default=0.7,
                        help="Saturation 증강 범위")
    parser.add_argument("--hsv-v",   type=float, default=0.4,
                        help="Value 증강 범위")
    parser.add_argument("--flipud",  type=float, default=0.0,
                        help="상하 뒤집기 확률")
    parser.add_argument("--fliplr",  type=float, default=0.5,
                        help="좌우 뒤집기 확률")
    parser.add_argument("--mosaic",  type=float, default=1.0,
                        help="모자이크 증강 확률")
    parser.add_argument("--mixup",   type=float, default=0.0,
                        help="MixUp 증강 확률")

    # ── 실행 환경 ────────────────────────────────────────────────
    parser.add_argument("--device",  type=str,   default="",
                        help="학습 장치: '0', '0,1', 'cpu' (비우면 자동 선택)")
    parser.add_argument("--workers", type=int,   default=4,
                        help="DataLoader 워커 수 (Windows는 0 권장)")
    parser.add_argument("--project", type=str,   default="runs/detect",
                        help="결과 저장 상위 디렉토리")
    parser.add_argument("--name",    type=str,   default="rps_finetune",
                        help="실험 이름")
    parser.add_argument("--resume",  action="store_true",
                        help="마지막 체크포인트부터 학습 재개")
    parser.add_argument("--patience",type=int,   default=20,
                        help="Early stopping patience (에폭 수)")
    parser.add_argument("--save-period", type=int, default=10,
                        help="N 에폭마다 체크포인트 저장 (-1: 비활성화)")

    return parser.parse_args()


def check_dataset(data_yaml: str):
    """데이터셋이 존재하는지 확인합니다."""
    import yaml
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg.get("path", "."))
    missing = []
    for split in ["train", "val"]:
        img_dir = root / cfg[split]
        if not img_dir.exists() or not any(img_dir.iterdir()):
            missing.append(str(img_dir))

    if missing:
        print("\n[경고] 아래 폴더에 이미지가 없습니다:")
        for p in missing:
            print(f"  - {p}")
        print("\n  데이터 준비 방법:")
        print("  1. python download_dataset.py --api-key YOUR_KEY")
        print("  2. 또는 data/train/images, data/valid/images에 직접 이미지를 넣으세요.")
        return False
    return True


def main():
    args = parse_args()

    # ── Ultralytics import ───────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics 패키지가 없습니다.")
        print("  pip install ultralytics 실행 후 재시도하세요.")
        raise

    # ── 데이터셋 확인 ────────────────────────────────────────────
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"데이터 yaml 파일을 찾을 수 없습니다: {data_path}")

    dataset_ready = check_dataset(str(data_path))
    if not dataset_ready:
        print("\n데이터셋을 먼저 준비하고 다시 실행하세요.")
        return

    # ── 모델 로드 ────────────────────────────────────────────────
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    print(f"\n{'='*60}")
    print(f"  모델   : {model_path}")
    print(f"  데이터 : {data_path}")
    print(f"  에폭   : {args.epochs}")
    print(f"  배치   : {args.batch}")
    print(f"  이미지 크기 : {args.imgsz}")
    print(f"  동결 레이어 : {args.freeze}")
    print(f"  장치   : {args.device if args.device else '자동'}")
    print(f"{'='*60}\n")

    model = YOLO(str(model_path))

    # ── 학습 실행 ────────────────────────────────────────────────
    results = model.train(
        data=str(data_path.resolve()),

        # 기본 설정
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,

        # 학습률
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,

        # 레이어 동결 (백본 보호, 헤드만 학습)
        freeze=args.freeze,

        # 증강
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,

        # 실행 환경
        device=args.device if args.device else None,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,

        # 저장/조기종료
        patience=args.patience,
        save_period=args.save_period,

        # 출력
        verbose=True,
        plots=True,        # 학습 곡선, 혼동행렬 등 시각화 저장
        val=True,          # 매 에폭마다 검증
        exist_ok=False,    # 같은 이름 실험 덮어쓰기 방지
    )

    # ── 학습 완료 요약 ───────────────────────────────────────────
    save_dir = Path(results.save_dir)
    best_pt  = save_dir / "weights" / "best.pt"

    print(f"\n{'='*60}")
    print(f"  학습 완료!")
    print(f"  결과 디렉토리 : {save_dir}")
    print(f"  최고 가중치   : {best_pt}")
    print(f"{'='*60}")
    print("\n추론 예시:")
    print(f"  python predict.py --model {best_pt} --source 0  # 웹캠")
    print(f"  python predict.py --model {best_pt} --source image.jpg")


if __name__ == "__main__":
    main()
