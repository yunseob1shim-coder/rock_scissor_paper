"""
quantize.py
-----------
ONNX 모델을 INT8 Static 양자화하고 FP32와 정확도를 비교합니다.

RPi5 (ARM Cortex-A76) 최적화:
  - FP16: HW 가속 없음 -> 비추천 (FP32보다 느릴 수 있음)
  - INT8 Dynamic: NEON 부분 활용, CNN에는 Static이 더 적합
  - INT8 Static: NEON UDOT/SDOT 완전 활용 -> 2-3x 속도 향상 [권장]

중요: 어떤 양자화도 수학적으로 완전한 정확도 보장은 불가능합니다.
      calibration 품질을 높여 손실을 최소화하는 것이 목표입니다.

사용법:
    # Step 1: FP32 ONNX export (export.py로 먼저 실행)
    python export.py --model runs/detect/rps_finetune/weights/best.pt --imgsz 320

    # Step 2: INT8 양자화 + 정확도 비교
    python quantize.py --model best.onnx --data data/rps.yaml

    # Step 3: 결과 확인 후 RPi5에 배포
    # best_int8.onnx 를 RPi5로 복사
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="INT8 Static Quantization for RPi5")
    parser.add_argument("--model",   type=str, required=True,
                        help="FP32 ONNX 모델 경로 (best.onnx)")
    parser.add_argument("--data",    type=str, default="data/rps.yaml",
                        help="데이터셋 yaml (calibration용 validation 이미지)")
    parser.add_argument("--output",  type=str, default="",
                        help="양자화 모델 저장 경로 (기본: best_int8.onnx)")
    parser.add_argument("--calib-method", type=str, default="Entropy",
                        choices=["MinMax", "Entropy", "Percentile"],
                        help="Calibration 방법 (Entropy: 정확도 최적, MinMax: 빠름)")
    parser.add_argument("--per-channel", action="store_true", default=True,
                        help="Per-channel 가중치 양자화 (정확도 향상, 권장)")
    parser.add_argument("--num-calib", type=int, default=100,
                        help="Calibration에 사용할 이미지 수 (많을수록 정확)")
    parser.add_argument("--imgsz",   type=int, default=320,
                        help="입력 이미지 크기 (export 시와 동일)")
    parser.add_argument("--conf",    type=float, default=0.25,
                        help="검출 신뢰도 임계값")
    parser.add_argument("--skip-compare", action="store_true",
                        help="FP32 vs INT8 정확도 비교 건너뛰기")
    return parser.parse_args()


# ── Calibration DataReader ────────────────────────────────────────

class CalibrationDataReader:
    """
    onnxruntime quantization용 Calibration 데이터 리더.
    validation 이미지를 letterbox 전처리 후 순서대로 제공합니다.
    """

    def __init__(self, image_dir: Path, imgsz: int, num_images: int, input_name: str):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        images = []
        for ext in exts:
            images.extend(image_dir.glob(ext))
        images = sorted(images)[:num_images]

        if not images:
            raise FileNotFoundError(
                f"Calibration 이미지 없음: {image_dir}\n"
                "  validation 이미지를 data/valid/images/ 에 배치하세요."
            )

        self.images = images
        self.imgsz  = imgsz
        self.input_name = input_name
        self._index = 0
        print(f"  Calibration 이미지: {len(images)}장 사용")

    def get_next(self):
        if self._index >= len(self.images):
            return None
        img_path = self.images[self._index]
        self._index += 1

        img = cv2.imread(str(img_path))
        if img is None:
            return self.get_next()

        tensor = _letterbox(img, self.imgsz)
        return {self.input_name: tensor}


def _letterbox(img_bgr: np.ndarray, imgsz: int) -> np.ndarray:
    """BGR 이미지 -> ONNX 입력 텐서 (FP32, NCHW, 0-1 정규화)"""
    h, w = img_bgr.shape[:2]
    scale = imgsz / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img_bgr, (nw, nh))
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    ph, pw = (imgsz - nh) // 2, (imgsz - nw) // 2
    canvas[ph:ph+nh, pw:pw+nw] = resized
    tensor = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(tensor, 0)


# ── 양자화 실행 ───────────────────────────────────────────────────

def run_quantization(fp32_path: Path, output_path: Path, calib_reader,
                     calib_method: str, per_channel: bool):
    """INT8 Static Quantization 실행"""
    try:
        from onnxruntime.quantization import (
            quantize_static, CalibrationMethod, QuantType, QuantFormat,
            quant_pre_process,
        )
    except ImportError:
        print("[ERROR] pip install onnxruntime 후 재시도하세요.")
        raise

    # Step 1: Pre-processing (shape inference + graph optimization)
    preprocessed_path = fp32_path.parent / (fp32_path.stem + "_preprocessed.onnx")
    print("\n[1/3] ONNX 전처리 (shape inference)...")
    quant_pre_process(
        input_model_path=str(fp32_path),
        output_model_path=str(preprocessed_path),
        skip_symbolic_shape=False,
    )

    # Step 2: Calibration method 매핑
    method_map = {
        "MinMax":     CalibrationMethod.MinMax,
        "Entropy":    CalibrationMethod.Entropy,
        "Percentile": CalibrationMethod.Percentile,
    }
    cal_method = method_map[calib_method]

    # Step 3: Static Quantization
    print(f"[2/3] INT8 Static Quantization ({calib_method}, per_channel={per_channel})...")
    quantize_static(
        model_input=str(preprocessed_path),
        model_output=str(output_path),
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,           # QDQ: ARM에서 성능 더 좋음
        activation_type=QuantType.QUInt8,       # 활성화: uint8
        weight_type=QuantType.QInt8,            # 가중치: int8
        per_channel=per_channel,                # per-channel: 정확도 향상
        calibrate_method=cal_method,
        extra_options={
            "ActivationSymmetric": False,        # 활성화: asymmetric (ReLU 이후 음수 없음)
            "WeightSymmetric": True,             # 가중치: symmetric (ARM 최적화)
            "AddQDQPairToWeight": True,
        },
    )

    # 임시 파일 삭제
    preprocessed_path.unlink(missing_ok=True)
    print(f"[3/3] 저장 완료: {output_path}")


# ── 정확도 비교 ───────────────────────────────────────────────────

def run_inference(session, images: list, imgsz: int, conf: float) -> tuple:
    """주어진 세션으로 이미지 목록을 추론하고 (검출 수, 평균 신뢰도, 평균 ms) 반환"""
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    total_det, total_conf, total_ms = 0, 0.0, 0.0

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        tensor = _letterbox(img, imgsz)

        t0 = time.perf_counter()
        raw = session.run([output_name], {input_name: tensor})[0]
        total_ms += (time.perf_counter() - t0) * 1000

        pred = raw[0]
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T
        nc = pred.shape[1] - 4
        scores = pred[:, 4:4+nc].max(axis=1)
        mask = scores >= conf
        total_det  += int(mask.sum())
        if mask.any():
            total_conf += float(scores[mask].mean())

    n = len(images)
    avg_ms   = total_ms / n if n else 0
    avg_conf = total_conf / n if n else 0
    return total_det, avg_conf, avg_ms


def compare_accuracy(fp32_path: Path, int8_path: Path,
                     val_dir: Path, imgsz: int, conf: float, num_images: int):
    """FP32 vs INT8 검출 결과를 비교합니다."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[SKIP] onnxruntime 없음. 비교를 건너뜁니다.")
        return

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    images = []
    for ext in exts:
        images.extend(val_dir.glob(ext))
    images = sorted(images)[:num_images]

    if not images:
        print(f"[SKIP] 비교용 이미지 없음: {val_dir}")
        return

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess_fp32 = ort.InferenceSession(str(fp32_path),  sess_options=sess_opts,
                                     providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(int8_path),  sess_options=sess_opts,
                                     providers=["CPUExecutionProvider"])

    print(f"\n[비교] {len(images)}장 이미지로 FP32 vs INT8 비교 중...")
    det32, conf32, ms32 = run_inference(sess_fp32, images, imgsz, conf)
    det8,  conf8,  ms8  = run_inference(sess_int8, images, imgsz, conf)

    size_fp32 = fp32_path.stat().st_size / 1024 / 1024
    size_int8 = int8_path.stat().st_size  / 1024 / 1024
    speedup   = ms32 / ms8 if ms8 > 0 else 0
    det_diff  = abs(det32 - det8) / max(det32, 1) * 100

    print(f"\n{'='*58}")
    print(f"  {'항목':18s}  {'FP32':>12s}  {'INT8':>12s}")
    print(f"  {'-'*54}")
    print(f"  {'모델 크기 (MB)':18s}  {size_fp32:>12.2f}  {size_int8:>12.2f}")
    print(f"  {'평균 추론 (ms)':18s}  {ms32:>12.1f}  {ms8:>12.1f}  ({speedup:.1f}x 빠름)")
    print(f"  {'총 검출 수':18s}  {det32:>12d}  {det8:>12d}")
    print(f"  {'평균 신뢰도':18s}  {conf32:>12.4f}  {conf8:>12.4f}")
    print(f"  {'검출 수 차이':18s}  {'':>12s}  {det_diff:>11.2f}%")
    print(f"{'='*58}")

    if det_diff <= 1.0 and abs(conf32 - conf8) <= 0.01:
        print("\n  [OK] 정확도 손실이 매우 낮습니다 (검출 차이 <1%, 신뢰도 차이 <0.01)")
        print("       INT8 모델을 RPi5 배포용으로 사용해도 안전합니다.")
    elif det_diff <= 5.0:
        print(f"\n  [주의] 검출 수 차이 {det_diff:.1f}%. 실제 영상으로 추가 검증 권장.")
    else:
        print(f"\n  [경고] 검출 수 차이 {det_diff:.1f}%. Calibration 이미지를 늘리거나")
        print("         --calib-method Entropy 로 재시도하거나 FP32 사용을 권장합니다.")

    return det_diff


# ── 메인 ─────────────────────────────────────────────────────────

def main():
    args = parse_args()

    try:
        import onnxruntime as ort
        from onnxruntime.quantization import quantize_static
    except ImportError:
        print("[ERROR] pip install onnxruntime 후 재시도하세요.")
        raise

    # 경로 설정
    fp32_path = Path(args.model)
    if not fp32_path.exists():
        raise FileNotFoundError(f"FP32 ONNX 모델 없음: {fp32_path}")

    output_path = Path(args.output) if args.output else \
                  fp32_path.parent / (fp32_path.stem + "_int8.onnx")

    # 데이터셋 yaml에서 val 경로 파싱
    import yaml
    with open(args.data, "r") as f:
        cfg = yaml.safe_load(f)
    data_root = Path(cfg.get("path", "."))
    val_dir   = data_root / cfg.get("val", "valid/images")

    # onnxruntime 세션으로 input name 획득
    sess_tmp = ort.InferenceSession(str(fp32_path),
                                    providers=["CPUExecutionProvider"])
    input_name = sess_tmp.get_inputs()[0].name
    del sess_tmp

    print(f"\n{'='*58}")
    print(f"  RPi5 INT8 Static Quantization")
    print(f"  FP32 모델   : {fp32_path}")
    print(f"  출력 모델   : {output_path}")
    print(f"  Calibration : {args.calib_method}, {args.num_calib}장")
    print(f"  Per-channel : {args.per_channel}")
    print(f"{'='*58}\n")

    # Calibration 데이터 준비
    calib_reader = CalibrationDataReader(
        image_dir=val_dir,
        imgsz=args.imgsz,
        num_images=args.num_calib,
        input_name=input_name,
    )

    # 양자화 실행
    run_quantization(
        fp32_path=fp32_path,
        output_path=output_path,
        calib_reader=calib_reader,
        calib_method=args.calib_method,
        per_channel=args.per_channel,
    )

    # 정확도 비교
    if not args.skip_compare:
        compare_accuracy(
            fp32_path=fp32_path,
            int8_path=output_path,
            val_dir=val_dir,
            imgsz=args.imgsz,
            conf=args.conf,
            num_images=args.num_calib,
        )

    print(f"\nRPi5 배포:")
    print(f"  pip install onnxruntime")
    print(f"  python predict_rpi.py --model {output_path.name} --source 0")


if __name__ == "__main__":
    main()
