"""
download_dataset.py
-------------------
로컬 zip 파일에서 Rock-Scissors-Paper 데이터셋을 추출하여
data/ 폴더에 배치합니다.

사용법:
    python download_dataset.py --local-zip rsp-brick-breaker/rock-paper-scissors.merged.yolo26.zip

    # 데이터셋 현황만 확인
    python download_dataset.py --check

zip 내부 구조 (자동 인식):
    train/images/, train/labels/
    valid/images/, valid/labels/
    test/images/,  test/labels/
"""

import argparse
import sys
import zipfile
from pathlib import Path


def extract_to_data(zip_path: Path):
    """zip을 메모리 효율적으로 읽어 data/ 폴더에 복사합니다."""
    if not zipfile.is_zipfile(zip_path):
        print(f"[ERROR] 유효한 zip 파일이 아닙니다: {zip_path}")
        sys.exit(1)

    dest_root = Path("data")
    total = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        entries = zf.namelist()
        for split in ["train", "valid", "test"]:
            for kind in ["images", "labels"]:
                prefix = f"{split}/{kind}/"
                files = [e for e in entries
                         if e.startswith(prefix) and not e.endswith("/")]
                if not files:
                    continue
                dst = dest_root / split / kind
                dst.mkdir(parents=True, exist_ok=True)
                for entry in files:
                    fname = Path(entry).name
                    (dst / fname).write_bytes(zf.read(entry))
                total += len(files)
                print(f"  [OK] {split}/{kind}: {len(files)}개")

    print(f"\n  총 {total}개 파일 복사 완료.")


def check_dataset():
    root = Path("data")
    for split in ["train", "valid", "test"]:
        imgs = list((root / split / "images").glob("*.*")) \
               if (root / split / "images").exists() else []
        lbls = list((root / split / "labels").glob("*.txt")) \
               if (root / split / "labels").exists() else []
        print(f"  {split:6s} : images={len(imgs):5d}  labels={len(lbls):5d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rock-Scissors-Paper 데이터셋 준비")
    parser.add_argument("--local-zip", type=str, default="",
                        help="로컬 zip 파일 경로")
    parser.add_argument("--check", action="store_true",
                        help="데이터셋 현황만 확인")
    args = parser.parse_args()

    print("=== 데이터셋 현황 ===")
    check_dataset()

    if args.check:
        sys.exit(0)

    if not args.local_zip:
        print("\n사용법:")
        print("  python download_dataset.py \\")
        print("    --local-zip rsp-brick-breaker/rock-paper-scissors.merged.yolo26.zip")
        sys.exit(0)

    zip_path = Path(args.local_zip)
    if not zip_path.exists():
        print(f"[ERROR] zip 파일 없음: {zip_path.resolve()}")
        sys.exit(1)

    print(f"\n=== zip 추출: {zip_path.name} ({zip_path.stat().st_size // 1024} KB) ===")
    extract_to_data(zip_path)

    print("\n=== 추출 후 현황 ===")
    check_dataset()
    print("\n학습 실행:")
    print("  python train.py --epochs 50 --batch 16")

    print("=== 데이터셋 현황 ===")
    check_dataset()

    if args.check:
        sys.exit(0)

    if not args.api_key:
        print("\n[INFO] --api-key 옵션을 추가하여 재실행하세요.")
        sys.exit(0)

    # 사용자가 명시적으로 지정했으면 해당 항목만, 아니면 후보 목록 순서대로 시도
    if args.workspace and args.project and args.version:
        candidates = [(args.workspace, args.project, args.version, "yolov8")]
    else:
        candidates = CANDIDATES

    print("\n=== Roboflow 다운로드 ===")

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "dataset.zip"
        success = False

        for ws, proj, ver, fmt in candidates:
            print(f"\n  시도: {ws}/{proj} v{ver} [{fmt}]")

            link = get_export_link(args.api_key, ws, proj, ver, fmt)
            if not link:
                print("  → export 링크 획득 실패, 다음 후보로")
                continue

            print(f"  링크: {link[:70]}...")

            # GCS 파일이 아직 없을 수 있으니 최대 3번 재시도
            for attempt in range(3):
                if attempt > 0:
                    wait = attempt * 10
                    print(f"  → GCS 대기 {wait}초 후 재시도...")
                    time.sleep(wait)
                    link = get_export_link(args.api_key, ws, proj, ver, fmt) or link

                print(f"  다운로드 중... (시도 {attempt+1}/3)")
                ok = try_download(link, zip_path)
                if ok:
                    print(f"  다운로드 성공: {zip_path.stat().st_size // 1024} KB")
                    success = True
                    break

            if success:
                break
            print("  → 다운로드 실패, 다음 후보로")

        if not success:
            print("\n[ERROR] 모든 후보 다운로드 실패.")
            print("  수동 다운로드: https://universe.roboflow.com/joseph-nelson/rock-paper-scissors-sxsw")
            print("  위 사이트에서 'YOLOv8' 포맷으로 다운로드 후 아래 경로에 배치:")
            print("    data/train/images/  data/train/labels/")
            print("    data/valid/images/  data/valid/labels/")
            sys.exit(1)

        print("  압축 해제 중...")
        extract_to_data(zip_path)

    print("\n=== 다운로드 후 현황 ===")
    check_dataset()


import argparse
import sys
import shutil
import zipfile
import tempfile
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("[ERROR] pip install requests")
    sys.exit(1)


# ── Roboflow API 설정 ─────────────────────────────────────────────
DEFAULT_WORKSPACE = "joseph-nelson"
DEFAULT_PROJECT   = "rock-paper-scissors-sxsw"
DEFAULT_VERSION   = 14


def get_export_link(api_key: str, workspace: str, project: str, version: int) -> str:
    """
    Roboflow API를 폴링하여 export 다운로드 링크를 반환합니다.

    API 동작:
      - export가 생성 중이면: {"ready": false, "progress": 0.5}
      - export 준비 완료이면: {"export": {"link": "https://...", "expires": 900}}
    """
    url = f"https://api.roboflow.com/{workspace}/{project}/{version}/yolov8"

    for attempt in range(60):  # 최대 2분 대기
        resp = requests.get(url, params={"api_key": api_key}, timeout=30)
        if resp.status_code != 200:
            print(f"\n[ERROR] API 오류 (HTTP {resp.status_code}): {resp.text[:300]}")
            sys.exit(1)

        data = resp.json()

        # export 준비 완료
        if "export" in data and "link" in data["export"]:
            if attempt > 0:
                print()  # 줄바꿈
            return data["export"]["link"]

        # export 생성 중
        if data.get("ready") is False:
            progress = data.get("progress", 0)
            print(f"\r  export 생성 중... {progress * 100:.0f}%  ({attempt * 2}s)", end="", flush=True)
            time.sleep(2)
            continue

        # 그 외 응답 (예: 빈 응답) → 잠시 후 재시도
        print(f"\r  응답 대기 중... ({attempt * 2}s)  ", end="", flush=True)
        time.sleep(2)

    print("\n[ERROR] export 준비 시간 초과. 잠시 후 다시 시도하세요.")
    sys.exit(1)


def download_zip(url: str, dest: Path) -> bool:
    """URL에서 zip 파일을 다운로드합니다. 성공 시 True, 404이면 False 반환."""
    resp = requests.get(url, stream=True, timeout=180, allow_redirects=True)
    if resp.status_code == 404:
        return False
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  {downloaded / total * 100:5.1f}%  ({downloaded // 1024} / {total // 1024} KB)",
                      end="", flush=True)
    print()
    return True


def extract_to_data(zip_path: Path):
    """zip을 압축 해제하여 data/ 폴더에 복사합니다."""
    extract_dir = zip_path.parent / "extracted"
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    dest_root = Path("data")
    for split in ["train", "valid", "test"]:
        for kind in ["images", "labels"]:
            # 압축 해제 구조가 다를 수 있으므로 재귀 탐색
            src = next((p for p in extract_dir.rglob(f"{split}/{kind}") if p.is_dir()), None)
            if not src:
                continue
            dst = dest_root / split / kind
            dst.mkdir(parents=True, exist_ok=True)
            files = [f for f in src.iterdir() if f.is_file()]
            for f in files:
                shutil.copy2(f, dst / f.name)
            if files:
                print(f"  [OK] {split}/{kind}: {len(files)}개")


def check_dataset():
    root = Path("data")
    for split in ["train", "valid", "test"]:
        imgs = list((root / split / "images").glob("*.*")) if (root / split / "images").exists() else []
        lbls = list((root / split / "labels").glob("*.txt")) if (root / split / "labels").exists() else []
        print(f"  {split:6s} : images={len(imgs):5d}  labels={len(lbls):5d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rock-Scissors-Paper 데이터셋 다운로드")
    parser.add_argument("--api-key",   type=str, default="")
    parser.add_argument("--workspace", type=str, default=DEFAULT_WORKSPACE)
    parser.add_argument("--project",   type=str, default=DEFAULT_PROJECT)
    parser.add_argument("--version",   type=int, default=DEFAULT_VERSION)
    parser.add_argument("--check",     action="store_true", help="현재 데이터셋 상태만 확인")
    args = parser.parse_args()

    print("=== 데이터셋 현황 ===")
    check_dataset()

    if args.check:
        sys.exit(0)

    if not args.api_key:
        print("\n[INFO] --api-key 옵션을 추가하여 재실행하세요.")
        sys.exit(0)

    print("\n=== Roboflow 다운로드 ===")
    print(f"  {args.workspace}/{args.project} v{args.version}")

    # 1. export 링크 획득 + 다운로드 (GCS 업로드 지연 대비 재시도)
    RETRY_DELAYS = [3, 5, 10, 15, 20, 30]
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "dataset.zip"
        downloaded = False
        for attempt, wait in enumerate(RETRY_DELAYS + [None]):
            # 매 시도마다 API에서 fresh signed URL 요청 (캐시된 만료 URL 방지)
            link = get_export_link(args.api_key, args.workspace, args.project, args.version)
            print(f"  [시도 {attempt+1}] {link[:70]}...")

            print("  다운로드 중...")
            ok = download_zip(link, zip_path)
            if ok:
                downloaded = True
                break

            if wait is None:
                print("[ERROR] 최대 재시도 횟수 초과.")
                sys.exit(1)
            print(f"  [404] GCS 업로드 대기 중... {wait}초 후 재시도")
            time.sleep(wait)

        print(f"  크기: {zip_path.stat().st_size // 1024} KB")

        if not zipfile.is_zipfile(zip_path):
            print("[ERROR] 유효한 zip 파일이 아닙니다. API 키와 버전 번호를 확인하세요.")
            sys.exit(1)

        print("  압축 해제 및 복사 중...")
        extract_to_data(zip_path)
    print("\n=== 다운로드 후 현황 ===")
    check_dataset()


import argparse
import sys
import shutil
import zipfile
import tempfile
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("[ERROR] requests 패키지가 없습니다. pip install requests 실행 후 재시도하세요.")
    sys.exit(1)


def _get_fresh_download_url(api_key: str, workspace: str, project: str, version: int) -> str:
    """Roboflow API에서 새 signed download URL을 요청합니다."""
    api_url = f"https://api.roboflow.com/{workspace}/{project}/{version}/yolov8"
    resp = requests.get(api_url, params={"api_key": api_key}, timeout=30)
    if resp.status_code != 200:
        print(f"[ERROR] API 요청 실패 (HTTP {resp.status_code}): {resp.text[:300]}")
        sys.exit(1)
    data = resp.json()
    if "export" not in data or "link" not in data["export"]:
        print(f"[ERROR] 응답에 download link 없음: {data}")
        sys.exit(1)
    return data["export"]["link"]


def download_from_roboflow(api_key: str, workspace: str, project: str, version: int):
    """Roboflow REST API를 직접 호출하여 데이터셋을 다운로드합니다.

    Roboflow는 export를 lazy하게 생성하므로, 첫 요청 시 GCS에 파일이
    아직 없을 수 있습니다. 재시도마다 fresh signed URL을 재요청합니다.
    """
    MAX_RETRIES = 6
    RETRY_DELAYS = [5, 10, 15, 20, 30, 30]  # 초 단위

    api_url = f"https://api.roboflow.com/{workspace}/{project}/{version}/yolov8"
    print(f"  API 요청: {api_url}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / "dataset.zip"

        for attempt in range(MAX_RETRIES):
            # 매 시도마다 fresh signed URL 재요청 (만료/미생성 대응)
            download_url = _get_fresh_download_url(api_key, workspace, project, version)
            print(f"  [시도 {attempt+1}/{MAX_RETRIES}] 다운로드 URL: {download_url[:70]}...")

            try:
                with requests.get(download_url, stream=True, timeout=180) as r:
                    if r.status_code == 404:
                        wait = RETRY_DELAYS[attempt]
                        print(f"  [404] export 생성 중... {wait}초 후 재시도")
                        time.sleep(wait)
                        continue
                    r.raise_for_status()

                    total = int(r.headers.get("content-length", 0))
                    downloaded = 0
                    with open(zip_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=65536):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                pct = downloaded / total * 100
                                print(f"\r  {pct:5.1f}%  ({downloaded//1024} KB / {total//1024} KB)",
                                      end="", flush=True)
                print(f"\n  다운로드 완료: {zip_path.stat().st_size // 1024} KB")
                break  # 성공

            except requests.HTTPError as e:
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_DELAYS[attempt]
                    print(f"\n  [오류] {e} — {wait}초 후 재시도...")
                    time.sleep(wait)
                else:
                    print(f"\n[ERROR] 다운로드 최종 실패: {e}")
                    sys.exit(1)
        else:
            print("[ERROR] 최대 재시도 횟수 초과. 잠시 후 다시 실행하거나 수동으로 데이터를 준비하세요.")
            sys.exit(1)

        # zip 검증 및 압축 해제
        if not zipfile.is_zipfile(zip_path):
            print("[ERROR] 다운로드된 파일이 유효한 zip 파일이 아닙니다.")
            print("        API 키 권한 또는 프로젝트/버전 번호를 확인하세요.")
            sys.exit(1)

        extract_dir = Path(tmp_dir) / "extracted"
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        print(f"  압축 해제 완료: {extract_dir}")

        # data/ 폴더에 복사
        dest_root = Path("data")
        for split in ["train", "valid", "test"]:
            # 압축 해제 후 구조가 다를 수 있으므로 재귀 탐색
            src_images = _find_dir(extract_dir, split, "images")
            src_labels = _find_dir(extract_dir, split, "labels")

            if src_images and src_images.exists():
                dst = dest_root / split / "images"
                dst.mkdir(parents=True, exist_ok=True)
                files = [f for f in src_images.iterdir() if f.is_file()]
                for f in files:
                    shutil.copy2(f, dst / f.name)
                print(f"  [OK] {split}/images: {len(files)}장 복사")

            if src_labels and src_labels.exists():
                dst = dest_root / split / "labels"
                dst.mkdir(parents=True, exist_ok=True)
                files = [f for f in src_labels.iterdir() if f.is_file()]
                for f in files:
                    shutil.copy2(f, dst / f.name)
                print(f"  [OK] {split}/labels: {len(files)}개 복사")

    print("\n[완료] 데이터셋 준비 완료!")


def _find_dir(root: Path, split: str, subdir: str) -> Path:
    """압축 해제 디렉토리에서 split/subdir 경로를 재귀적으로 탐색합니다."""
    # 직접 경로 시도
    direct = root / split / subdir
    if direct.exists():
        return direct
    # 하위 폴더에 있는 경우
    for p in root.rglob(f"{split}/{subdir}"):
        if p.is_dir():
            return p
    return None


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
