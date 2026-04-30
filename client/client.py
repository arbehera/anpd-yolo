"""
ANPR Python Client
==================

Client-side script that handles:
  1. Image validation (type, size, dimensions)
  2. Client-side resize (>1920px → 1920px via OpenCV)
  3. Upload to the ANPR server with progress tracking
  4. Result display with annotated image output
  5. Manual plate validation (offline regex check)
  6. Batch processing (directory of images)

Usage:
  # Single image
  python client.py img1.jpg

  # Multiple images
  python client.py img1.jpg img2.jpg img3.jpg

  # Entire directory
  python client.py --dir ./plates/

  # Custom server
  python client.py img1.jpg --server http://192.168.1.100:8000

  # Just validate a plate string (no server needed)
  python client.py --validate KA01AB1234

  # Save annotated result image
  python client.py img1.jpg --save-annotated

  # Detection only (no OCR)
  python client.py img1.jpg --detect-only

  # JSON output (for piping to other tools)
  python client.py img1.jpg --json
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import requests
    from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
    HAS_TOOLBELT = True
except ImportError:
    import requests
    HAS_TOOLBELT = False


# ---------------------------------------------------------------------------
# Client-side config (mirrors server limits)
# ---------------------------------------------------------------------------

DEFAULT_SERVER = "http://localhost:8000"
MAX_FILE_MB = 10
MAX_DIMENSION = 1920
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

# Client-side plate patterns (same regex as server, for offline validation)
PLATE_PATTERNS = {
    "private": re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$"),
    "bh":      re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{2}$"),
    "gov":     re.compile(r"^[A-Z]{2}[0-9]{1,2}G[A-Z]{0,2}[0-9]{4}$"),
    "army":    re.compile(r"^[0-9]{2}[A-Z][0-9]{5,6}[A-Z]$"),
}

# ---------------------------------------------------------------------------
# Colors for terminal output
# ---------------------------------------------------------------------------

class C:
    """ANSI color codes. Disabled on non-TTY or Windows without ANSI support."""
    _enabled = sys.stdout.isatty() and os.name != 'nt'

    RESET  = '\033[0m'  if _enabled else ''
    BOLD   = '\033[1m'  if _enabled else ''
    DIM    = '\033[2m'  if _enabled else ''
    GREEN  = '\033[92m' if _enabled else ''
    RED    = '\033[91m' if _enabled else ''
    YELLOW = '\033[93m' if _enabled else ''
    CYAN   = '\033[96m' if _enabled else ''
    WHITE  = '\033[97m' if _enabled else ''


# ---------------------------------------------------------------------------
# Step 1-2: Client-side validation
# ---------------------------------------------------------------------------

def validate_file(path: Path) -> tuple[bool, str]:
    """Validate file before any processing. Returns (ok, message)."""

    if not path.exists():
        return False, f"File not found: {path}"

    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported format: {path.suffix} (use JPEG, PNG, or WebP)"

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        return False, f"File too large: {size_mb:.1f} MB (max {MAX_FILE_MB} MB)"

    return True, "OK"


def validate_image(image: np.ndarray) -> tuple[bool, str]:
    """Validate decoded image dimensions."""
    h, w = image.shape[:2]
    if max(h, w) > 4000:
        return False, f"Image too large: {w}x{h}px (max 4000px before resize)"
    if min(h, w) < 50:
        return False, f"Image too small: {w}x{h}px (min 50px)"
    return True, "OK"


# ---------------------------------------------------------------------------
# Step 3: Client-side resize
# ---------------------------------------------------------------------------

def resize_if_needed(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Resize image if larger than MAX_DIMENSION. Returns (image, was_resized)."""
    h, w = image.shape[:2]
    if max(h, w) <= MAX_DIMENSION:
        return image, False

    scale = MAX_DIMENSION / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, True


def encode_to_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    """Encode numpy array to JPEG bytes."""
    _, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Step 4: Upload with progress
# ---------------------------------------------------------------------------

def make_progress_callback(filename: str, total_size: int):
    """Create a progress callback that prints an upload progress bar."""
    start_time = time.time()
    last_print = [0]

    def callback(monitor):
        now = time.time()
        # Throttle updates to every 100ms
        if now - last_print[0] < 0.1 and monitor.bytes_read < total_size:
            return
        last_print[0] = now

        pct = min(monitor.bytes_read / total_size * 100, 100)
        elapsed = now - start_time
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = '█' * filled + '░' * (bar_len - filled)

        speed = ""
        if elapsed > 0 and monitor.bytes_read > 0:
            kbps = (monitor.bytes_read / 1024) / elapsed
            speed = f" {kbps:.0f} KB/s"

        sys.stdout.write(f"\r  {C.DIM}[{bar}] {pct:5.1f}%{speed}{C.RESET}")
        sys.stdout.flush()

        if monitor.bytes_read >= total_size:
            sys.stdout.write(f"\r  {C.DIM}[{'█' * bar_len}] 100.0%  Processing on server...{C.RESET}")
            sys.stdout.flush()

    return callback


def upload_image(server: str, endpoint: str, image_bytes: bytes,
                 filename: str = "plate.jpg", show_progress: bool = True) -> dict:
    """Upload image to the ANPR server and return the JSON response."""
    url = f"{server.rstrip('/')}/{endpoint.lstrip('/')}"

    if HAS_TOOLBELT and show_progress:
        # Upload with progress bar
        encoder = MultipartEncoder(
            fields={'file': (filename, image_bytes, 'image/jpeg')}
        )
        callback = make_progress_callback(filename, len(image_bytes))
        monitor = MultipartEncoderMonitor(encoder, callback)

        start = time.time()
        resp = requests.post(url, data=monitor,
                             headers={'Content-Type': monitor.content_type},
                             timeout=60)
        elapsed = time.time() - start

        if show_progress:
            sys.stdout.write(f"\r  {C.GREEN}[{'█' * 30}] Done in {elapsed:.1f}s{' ' * 20}{C.RESET}\n")
            sys.stdout.flush()
    else:
        # Simple upload without progress
        start = time.time()
        files = {'file': (filename, image_bytes, 'image/jpeg')}
        resp = requests.post(url, files=files, timeout=60)
        elapsed = time.time() - start

    if resp.status_code != 200:
        try:
            detail = resp.json().get('detail', resp.text)
        except Exception:
            detail = resp.text
        raise RuntimeError(f"Server error {resp.status_code}: {detail}")

    result = resp.json()
    result['_elapsed'] = round(elapsed, 2)
    return result


# ---------------------------------------------------------------------------
# Client-side plate validation (no server needed)
# ---------------------------------------------------------------------------

def validate_plate_offline(plate: str) -> dict:
    """Validate a plate string using client-side regex patterns."""
    normalized = re.sub(r'[^A-Z0-9]', '', plate.upper().strip())

    matched = None
    for name, pattern in PLATE_PATTERNS.items():
        if pattern.match(normalized):
            matched = name
            break

    return {
        "input": plate,
        "normalized": normalized,
        "valid_format": matched is not None,
        "matched_pattern": matched,
    }


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------

def print_result(result: dict, source: str = ""):
    """Print recognition result in a formatted table."""
    text = result.get('text', '')
    conf = result.get('confidence', 0)
    stab = result.get('stability', 0)
    valid = result.get('valid_format', False)
    elapsed = result.get('_elapsed', 0)
    dets = result.get('detections', [])

    print()
    print(f"  {C.DIM}{'─' * 52}{C.RESET}")
    if source:
        print(f"  {C.DIM}Source:     {source}{C.RESET}")

    if result.get('success'):
        plate_color = C.GREEN if valid else C.YELLOW
        print(f"  {C.BOLD}Plate:     {plate_color}{text}{C.RESET}")
        print(f"  Confidence: {conf * 100:.1f}%")
        print(f"  Stability:  {stab * 100:.1f}%")

        valid_str = f"{C.GREEN}VALID{C.RESET}" if valid else f"{C.RED}INVALID{C.RESET}"
        print(f"  Format:     {valid_str}")

        if dets:
            print(f"  Detections: {len(dets)} plate(s) found")
            for i, d in enumerate(dets):
                print(f"    {C.DIM}[{i}] ({d['x1']},{d['y1']})→({d['x2']},{d['y2']}) "
                      f"conf={d['confidence']:.2f}{C.RESET}")
    else:
        print(f"  {C.RED}No plate recognized{C.RESET}")

    if elapsed:
        print(f"  {C.DIM}Time:      {elapsed}s{C.RESET}")
    print(f"  {C.DIM}{'─' * 52}{C.RESET}")


def print_validation(result: dict):
    """Print offline validation result."""
    valid = result['valid_format']
    color = C.GREEN if valid else C.RED
    status = "VALID" if valid else "INVALID"

    print()
    print(f"  {C.DIM}{'─' * 40}{C.RESET}")
    print(f"  Input:      {result['input']}")
    print(f"  Normalized: {result['normalized']}")
    print(f"  Format:     {color}{status}{C.RESET}")
    if result['matched_pattern']:
        print(f"  Pattern:    {result['matched_pattern']}")
    print(f"  {C.DIM}{'─' * 40}{C.RESET}")


# ---------------------------------------------------------------------------
# Annotated image output
# ---------------------------------------------------------------------------

def save_annotated(image: np.ndarray, result: dict, output_path: Path):
    """Draw bounding boxes and plate text on the image, save to disk."""
    annotated = image.copy()

    dets = result.get('detections', [])
    text = result.get('text', '')

    for i, det in enumerate(dets):
        x1, y1 = det['x1'], det['y1']
        x2, y2 = det['x2'], det['y2']
        color = (0, 230, 118) if i == 0 else (0, 193, 255)  # Green for best, yellow for others

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        if i == 0 and text:
            # Label background
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            label_y = y1 - 10 if y1 > 30 else y2 + 25
            cv2.rectangle(annotated, (x1, label_y - th - 6), (x1 + tw + 10, label_y + 4), color, -1)
            cv2.putText(annotated, text, (x1 + 5, label_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.imwrite(str(output_path), annotated)
    print(f"  {C.DIM}Annotated image saved: {output_path}{C.RESET}")


# ---------------------------------------------------------------------------
# Server health check
# ---------------------------------------------------------------------------

def check_server(server: str) -> bool:
    """Check if the ANPR server is reachable."""
    try:
        resp = requests.get(f"{server}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            paddle = "✓" if data.get('paddle_available') else "✗"
            easy = "✓" if data.get('easyocr_available') else "✗"
            codes = data.get('state_codes_loaded', 0)
            print(f"  {C.GREEN}Server online{C.RESET} at {server}")
            print(f"  {C.DIM}PaddleOCR: {paddle}  EasyOCR: {easy}  "
                  f"State codes: {codes}{C.RESET}")
            return True
    except requests.ConnectionError:
        pass
    except Exception as e:
        print(f"  {C.RED}Server error: {e}{C.RESET}")

    print(f"  {C.RED}Server offline{C.RESET} at {server}")
    return False


# ---------------------------------------------------------------------------
# Process a single image (Steps 1-4 + display)
# ---------------------------------------------------------------------------

def process_image(image_path: Path, server: str, args) -> Optional[dict]:
    """Full client-side flow for one image."""

    print(f"\n{C.CYAN}Processing: {image_path.name}{C.RESET}")

    # Step 1: Validate file
    ok, msg = validate_file(image_path)
    if not ok:
        print(f"  {C.RED}Rejected: {msg}{C.RESET}")
        return None

    # Step 2: Read and validate image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  {C.RED}Cannot decode image{C.RESET}")
        return None

    ok, msg = validate_image(image)
    if not ok:
        print(f"  {C.RED}Rejected: {msg}{C.RESET}")
        return None

    orig_h, orig_w = image.shape[:2]
    orig_size = image_path.stat().st_size

    # Step 3: Resize if needed
    image, was_resized = resize_if_needed(image)
    new_h, new_w = image.shape[:2]

    if was_resized:
        print(f"  {C.DIM}Resized: {orig_w}×{orig_h} → {new_w}×{new_h}{C.RESET}")

    # Encode to JPEG for upload
    jpeg_bytes = encode_to_jpeg(image)
    upload_kb = len(jpeg_bytes) / 1024

    print(f"  {C.DIM}Upload size: {upload_kb:.0f} KB "
          f"(original: {orig_size / 1024:.0f} KB){C.RESET}")

    # Step 4: Upload to server
    endpoint = "detect" if args.detect_only else "recognize"
    try:
        result = upload_image(server, endpoint, jpeg_bytes,
                              filename=image_path.name,
                              show_progress=not args.json)
    except RuntimeError as e:
        print(f"  {C.RED}{e}{C.RESET}")
        return None
    except requests.ConnectionError:
        print(f"  {C.RED}Cannot connect to server at {server}{C.RESET}")
        return None

    # Display result
    if args.json:
        result['source'] = str(image_path)
        print(json.dumps(result, ensure_ascii=False))
    else:
        print_result(result, source=str(image_path))

    # Save annotated image if requested
    if args.save_annotated and result.get('success'):
        out_name = image_path.stem + "_result" + image_path.suffix
        out_path = Path(args.output_dir) / out_name
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        save_annotated(image, result, out_path)

    return result


# ---------------------------------------------------------------------------
# Batch processing summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]):
    """Print summary statistics for batch processing."""
    total = len(results)
    successful = sum(1 for r in results if r and r.get('success'))
    valid = sum(1 for r in results if r and r.get('valid_format'))
    failed = total - successful

    avg_conf = 0
    confs = [r['confidence'] for r in results if r and r.get('success')]
    if confs:
        avg_conf = sum(confs) / len(confs)

    avg_time = 0
    times = [r['_elapsed'] for r in results if r and '_elapsed' in r]
    if times:
        avg_time = sum(times) / len(times)

    print(f"\n{'=' * 52}")
    print(f"  {C.BOLD}Batch Summary{C.RESET}")
    print(f"  Total:      {total}")
    print(f"  Recognized: {C.GREEN}{successful}{C.RESET}")
    print(f"  Valid:       {C.GREEN}{valid}{C.RESET}")
    print(f"  Failed:     {C.RED}{failed}{C.RESET}")
    print(f"  Avg conf:   {avg_conf * 100:.1f}%")
    print(f"  Avg time:   {avg_time:.2f}s")
    print(f"{'=' * 52}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="ANPR Python Client — recognize Indian license plates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s img1.jpg                          Single image
  %(prog)s img1.jpg img2.jpg                 Multiple images
  %(prog)s --dir ./plates/                   Entire directory
  %(prog)s img1.jpg --server http://x:8000   Custom server
  %(prog)s --validate KA01AB1234             Offline validation
  %(prog)s img1.jpg --save-annotated         Save result image
  %(prog)s img1.jpg --detect-only            Bounding boxes only
  %(prog)s img1.jpg --json                   Machine-readable output
  %(prog)s --health                          Check server status
        """,
    )

    parser.add_argument("images", nargs="*", type=Path, help="Image file(s) to process")
    parser.add_argument("--server", default=DEFAULT_SERVER,
                        help=f"Server URL (default: {DEFAULT_SERVER})")
    parser.add_argument("--dir", type=Path, help="Process all images in a directory")
    parser.add_argument("--validate", type=str, metavar="PLATE",
                        help="Validate a plate string offline (no server needed)")
    parser.add_argument("--detect-only", action="store_true",
                        help="Run detection only (no OCR)")
    parser.add_argument("--save-annotated", action="store_true",
                        help="Save annotated result images")
    parser.add_argument("--output-dir", default="results",
                        help="Directory for annotated images (default: results)")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON (for piping)")
    parser.add_argument("--health", action="store_true",
                        help="Check if the server is online")
    parser.add_argument("--no-resize", action="store_true",
                        help="Skip client-side resize (send original)")

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Offline validation (no server) ---
    if args.validate:
        result = validate_plate_offline(args.validate)
        if args.json:
            print(json.dumps(result))
        else:
            print_validation(result)
        return

    # --- Health check ---
    if args.health:
        check_server(args.server)
        return

    # --- Collect images ---
    images = list(args.images) if args.images else []

    if args.dir:
        if not args.dir.is_dir():
            print(f"{C.RED}Not a directory: {args.dir}{C.RESET}")
            sys.exit(1)
        for ext in ALLOWED_EXTENSIONS:
            images.extend(sorted(args.dir.glob(f"*{ext}")))
            images.extend(sorted(args.dir.glob(f"*{ext.upper()}")))

    if not images:
        print(f"{C.RED}No images specified. Use --help for usage.{C.RESET}")
        sys.exit(1)

    # Remove duplicates, preserve order
    seen = set()
    unique = []
    for img in images:
        resolved = img.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(img)
    images = unique

    # Override resize if flag set
    global MAX_DIMENSION
    if args.no_resize:
        MAX_DIMENSION = 99999

    # --- Check server ---
    if not args.json:
        print(f"\n{C.BOLD}ANPR Client{C.RESET}")
        print(f"  {C.DIM}Server: {args.server}{C.RESET}")
        print(f"  {C.DIM}Images: {len(images)}{C.RESET}")

    if not check_server(args.server):
        sys.exit(1)

    # --- Process ---
    results = []
    for image_path in images:
        result = process_image(image_path, args.server, args)
        results.append(result)

    # --- Summary for batch ---
    if len(images) > 1 and not args.json:
        print_summary(results)


if __name__ == "__main__":
    main()
