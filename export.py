"""
export.py
---------
학습된 best.pt 모델을 Raspberry Pi 5 배포용 ONNX 포맷으로 변환합니다.

RPi5 권장 형식: ONNX (onnxruntime-aarch64, ~10MB)
  - PyTorch(.pt)  : PyTorch ARM 빌드 필요 (무겁고 느림)
  - ONNX(.onnx)   : onnxruntime만 필요, CPU 최적화, 빠름  <-- 권장
  - NCNN          : 가장 빠르지만 빌드 복잡
  - TFLite        : TensorFlow Lite 필요

사용법:
    python export.py --model runs/detect/rps_finetune/weights/best.pt
    python export.py --model runs/detect/rps_finetune/weights/best.pt --imgsz 320

RPi5에서 추론:
    pip install onnxruntime   # PC/RPi5 공통
    python predict_rpi.py --model best.onnx --source 0
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="ONNX export for Raspberry Pi 5")
    parser.add_argument("--model",   type=str, required=True,
                        help="학습된 가중치 경로 (best.pt)")
    parser.add_argument("--imgsz",   type=int, default=320,
                        help="입력 이미지 크기 (RPi5 권장: 320, 640은 4x 느림)")
    parser.add_argument("--format",  type=str, default="onnx",
                        choices=["onnx", "ncnn", "tflite", "torchscript"],
                        help="출력 포맷 (기본: onnx)")
    parser.add_argument("--half",    action="store_true",
                        help="FP16 (RPi5 Cortex-A76에는 FP16 HW 가속 없음 -> 비추천)")
    parser.add_argument("--int8",    action="store_true",
                        help="INT8 양자화 (정확도 손실 가능, quantize.py 사용 권장)")
    parser.add_argument("--simplify",action="store_true", default=True,
                        help="ONNX 그래프 단순화 (onnx-simplifier 설치 시)")
    parser.add_argument("--opset",   type=int, default=12,
                        help="ONNX opset 버전 (RPi5 onnxruntime 호환: 11~17)")
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

    print(f"\n{'='*55}")
    print(f"  입력 모델  : {model_path}")
    print(f"  출력 포맷  : {args.format.upper()}")
    print(f"  이미지 크기: {args.imgsz}x{args.imgsz}")
    print(f"  INT8 양자화: {'ON' if args.int8 else 'OFF'}")
    if args.half:
        print("  [주의] RPi5 Cortex-A76은 FP16 HW 가속 없음 -> FP32보다 느릴 수 있음")
    print(f"{'='*55}\n")

    model = YOLO(str(model_path))

    export_path = model.export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        simplify=args.simplify,
        opset=args.opset,
        device="cpu",   # RPi5 배포용이므로 CPU 기준으로 export
    )

    out = Path(export_path)
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"\n{'='*55}")
    print(f"  변환 완료!")
    print(f"  출력 파일  : {out}")
    print(f"  파일 크기  : {size_mb:.2f} MB")
    print(f"{'='*55}")
    print(f"\nRPi5 배포 방법:")
    print(f"  1. {out.name} 파일을 RPi5로 복사")
    print(f"  2. RPi5에서: pip install onnxruntime")
    print(f"  3. RPi5에서: python predict_rpi.py --model {out.name} --source 0")


if __name__ == "__main__":
    main()
