"""
predict_rpi.py
--------------
Raspberry Pi 5 전용 추론 스크립트 (ONNX Runtime 사용)
PyTorch 없이 onnxruntime만으로 동작합니다.

RPi5 설치:
    pip install onnxruntime opencv-python numpy

사용법:
    python predict_rpi.py --model best.onnx --source 0          # 웹캠
    python predict_rpi.py --model best.onnx --source image.jpg  # 이미지
    python predict_rpi.py --model best.onnx --source video.mp4  # 영상
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# 클래스 이름 (rps.yaml과 동일 순서)
CLASS_NAMES = {0: "rock", 1: "scissors", 2: "paper"}
CLASS_COLORS = {0: (0, 80, 255), 1: (0, 220, 50), 2: (255, 140, 0)}


def parse_args():
    parser = argparse.ArgumentParser(description="RPi5 ONNX 추론")
    parser.add_argument("--model",  type=str, required=True,
                        help="ONNX 모델 경로 (best.onnx)")
    parser.add_argument("--source", type=str, default="0",
                        help="입력 소스: 웹캠 번호(0), 이미지, 영상 경로")
    parser.add_argument("--imgsz",  type=int, default=320,
                        help="추론 이미지 크기 (export 시 사용한 크기와 동일하게)")
    parser.add_argument("--conf",   type=float, default=0.4,
                        help="최소 신뢰도 임계값")
    parser.add_argument("--iou",    type=float, default=0.45,
                        help="NMS IoU 임계값")
    parser.add_argument("--no-show",action="store_true",
                        help="화면 출력 비활성화 (헤드리스 환경)")
    parser.add_argument("--save",   type=str, default="",
                        help="결과 영상 저장 경로 (비우면 저장 안 함)")
    return parser.parse_args()


# ── 전처리 / 후처리 ──────────────────────────────────────────────

def preprocess(img_bgr: np.ndarray, imgsz: int):
    """BGR 이미지를 ONNX 입력 텐서로 변환합니다."""
    h, w = img_bgr.shape[:2]
    # letterbox 리사이즈
    scale = imgsz / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img_bgr, (nw, nh))

    # 패딩
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    pad_y = (imgsz - nh) // 2
    pad_x = (imgsz - nw) // 2
    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized

    # HWC→CHW, BGR→RGB, 정규화
    tensor = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, 0)  # (1, 3, H, W)
    return tensor, scale, pad_x, pad_y


def nms(boxes, scores, iou_threshold):
    """간단한 NMS 구현 (torchvision 없이)."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_threshold]
    return keep


def postprocess(output: np.ndarray, orig_h, orig_w,
                scale, pad_x, pad_y, imgsz, conf_thr, iou_thr):
    """
    YOLO ONNX 출력 (1, 7, N) 또는 (1, N, 7) 형태를 바운딩박스로 변환합니다.
    출력 형태: [x_center, y_center, w, h, cls0_conf, cls1_conf, cls2_conf, ...]
    """
    pred = output[0]  # (7, N) or (N, 7)
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T  # → (N, 7+)

    nc = pred.shape[1] - 4
    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:4+nc]

    class_ids = class_scores.argmax(axis=1)
    confs = class_scores.max(axis=1)

    mask = confs >= conf_thr
    if not mask.any():
        return []

    boxes_xywh = boxes_xywh[mask]
    class_ids  = class_ids[mask]
    confs      = confs[mask]

    # xywh → xyxy (모델 출력 공간)
    bx, by, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = bx - bw / 2
    y1 = by - bh / 2
    x2 = bx + bw / 2
    y2 = by + bh / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    keep = nms(boxes_xyxy, confs, iou_thr)
    results = []
    for idx in keep:
        # letterbox 역변환 → 원본 이미지 좌표
        ox1 = (boxes_xyxy[idx, 0] - pad_x) / scale
        oy1 = (boxes_xyxy[idx, 1] - pad_y) / scale
        ox2 = (boxes_xyxy[idx, 2] - pad_x) / scale
        oy2 = (boxes_xyxy[idx, 3] - pad_y) / scale

        ox1 = max(0, min(ox1, orig_w))
        oy1 = max(0, min(oy1, orig_h))
        ox2 = max(0, min(ox2, orig_w))
        oy2 = max(0, min(oy2, orig_h))

        results.append({
            "box":   [int(ox1), int(oy1), int(ox2), int(oy2)],
            "conf":  float(confs[idx]),
            "class": int(class_ids[idx]),
            "label": CLASS_NAMES.get(int(class_ids[idx]), str(class_ids[idx])),
        })
    return results


def draw(frame, detections):
    """검출 결과를 프레임에 그립니다."""
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        color = CLASS_COLORS.get(d["class"], (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{d['label']} {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame


# ── 메인 ─────────────────────────────────────────────────────────

def main():
    args = parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        print("[ERROR] onnxruntime 없음. 아래 명령어로 설치하세요:")
        print("  pip install onnxruntime")
        raise

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일 없음: {model_path}")

    # ONNX 세션 생성 (RPi5: CPUExecutionProvider)
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4   # RPi5 코어 수
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"[OK] 모델 로드 완료: {model_path.name}")
    print(f"     입력: {session.get_inputs()[0].shape}")
    print(f"     출력: {session.get_outputs()[0].shape}")

    # 소스 판단
    source = args.source
    is_webcam = source.isdigit()
    cap = None
    writer = None

    if is_webcam or source.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        cap = cv2.VideoCapture(int(source) if is_webcam else source)
        if not cap.isOpened():
            raise RuntimeError(f"영상 소스를 열 수 없습니다: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        if args.save:
            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(
                args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))

    frame_count, total_ms = 0, 0.0
    try:
        if cap is not None:
            # 영상 / 웹캠 루프
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                t0 = time.perf_counter()
                tensor, scale, px, py = preprocess(frame, args.imgsz)
                raw_out = session.run([output_name], {input_name: tensor})
                detections = postprocess(
                    raw_out[0], frame.shape[0], frame.shape[1],
                    scale, px, py, args.imgsz, args.conf, args.iou)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                total_ms += elapsed_ms
                frame_count += 1

                frame = draw(frame, detections)
                fps_text = f"{1000/elapsed_ms:.1f} FPS"
                cv2.putText(frame, fps_text, (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if writer:
                    writer.write(frame)
                if not args.no_show:
                    cv2.imshow("Rock-Scissors-Paper (RPi5)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        else:
            # 단일 이미지
            img_path = Path(source)
            if img_path.is_dir():
                images = list(img_path.glob("*.jpg")) + list(img_path.glob("*.png"))
            else:
                images = [img_path]

            for img_p in images:
                frame = cv2.imread(str(img_p))
                if frame is None:
                    print(f"[SKIP] 읽기 실패: {img_p}")
                    continue

                t0 = time.perf_counter()
                tensor, scale, px, py = preprocess(frame, args.imgsz)
                raw_out = session.run([output_name], {input_name: tensor})
                detections = postprocess(
                    raw_out[0], frame.shape[0], frame.shape[1],
                    scale, px, py, args.imgsz, args.conf, args.iou)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                frame_count += 1
                total_ms += elapsed_ms

                frame = draw(frame, detections)
                print(f"  {img_p.name}: {len(detections)}개 검출 ({elapsed_ms:.1f} ms)")

                if args.save:
                    out_p = Path(args.save)
                    out_p.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_p / img_p.name), frame)
                if not args.no_show:
                    cv2.imshow(img_p.name, frame)
                    cv2.waitKey(0)

    finally:
        if cap:
            cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    if frame_count > 0:
        avg_ms = total_ms / frame_count
        print(f"\n평균 추론 시간: {avg_ms:.1f} ms / frame  ({1000/avg_ms:.1f} FPS)")


if __name__ == "__main__":
    main()
