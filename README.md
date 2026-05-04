# Indian ANPR — License Plate Recognition API

Python client + FastAPI server for Indian license plate recognition.
Dual OCR (PaddleOCR 2.9.1 + EasyOCR) with YOLOv8 detection.

## Project Structure

```text
D:\LP_API\
├── server/
│   ├── __init__.py             (empty file)
│   ├── app.py                  FastAPI server (endpoints, middleware, CORS)
│   └── anpr_pipeline.py        Original ANPR pipeline (detection + OCR + validation)
│
├── client/
│   └── client.py               Python client (validate, resize, upload, display)
│
├── test_images/                Your test plate images
│
├── lp_model.pt                 YOLOv8 plate detection model
├── run.py                      Server entry point
├── requirements.txt            Python dependencies
├── Dockerfile                  Docker container build file
└── README.md                   This file
```

## Dependencies

```text
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.9
pydantic>=2.0.0
opencv-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.0.0
easyocr>=1.7.0
requests>=2.31.0
paddlepaddle==2.6.2
paddleocr==2.9.1
```

**Important version notes:**

- PaddleOCR must be **2.9.1** (not 3.x — the v3 API is completely different and broken on Windows)
- PaddlePaddle must be **2.6.2** (matches PaddleOCR 2.9.1)

## Quick Start

### 1. Install dependencies

```Download `lp_model.pt` from the [Releases page](https://github.com/arbehera/anpd-yolo/releases/latest) and place it in the project root.```

```powershell
cd D:\LP_API
pip install -r requirements.txt
```

### 2. Place your YOLO model

Copy `lp_model.pt` to `D:\LP_API\`.

### 3. Start the server (Terminal 1)

```powershell
python run.py
```

Wait until you see:

```text
ANPR pipeline ready.
Uvicorn running on http://0.0.0.0:8000
```

### 4. Test with client (Terminal 2)

Open a second PowerShell, activate your venv, then:

```powershell
cd D:\LP_API

# Check server health
python client/client.py --health

# Single image
python client/client.py test_images\img4.1.jpg

# All images in a folder
python client/client.py --dir test_images

# Validate a plate string (no server needed)
python client/client.py --validate KA01AB1234
```

### 5. Test with Swagger UI

Open browser: `http://localhost:8000/docs`

1. Click **POST /recognize**
2. Click **Try it out**
3. Click **Choose File** — select a plate image
4. Click **Execute**
5. Scroll down to see the result

## API Endpoints

| Method | Endpoint     | Purpose                                  |
|--------|--------------|------------------------------------------|
| GET    | /health      | Server status (OCR engines, state codes) |
| POST   | /recognize   | Full pipeline: image → plate result      |
| POST   | /detect      | Detection only: image → bounding boxes   |
| GET    | /validate    | Validate plate string (no image needed)  |
| GET    | /state-codes | List all valid RTO state codes           |

## Client Usage

```powershell
# Single image
python client/client.py img1.jpg

# Multiple images
python client/client.py img1.jpg img2.jpg img3.jpg

# Entire folder
python client/client.py --dir test_images

# Custom server URL
python client/client.py img1.jpg --server http://192.168.1.100:8000

# Save annotated images with bounding boxes
python client/client.py img1.jpg --save-annotated --output-dir results

# Detection only (no OCR)
python client/client.py img1.jpg --detect-only

# JSON output (for scripting)
python client/client.py img1.jpg --json

# Offline plate validation (no server needed)
python client/client.py --validate KA01AB1234

# Skip client-side resize
python client/client.py img1.jpg --no-resize
```

## What Runs Where

| Step | Where  | What Happens                                         |
|------|--------|------------------------------------------------------|
| 1    | Client | Validate file type (JPEG/PNG/WebP) and size (<10 MB) |
| 2    | Client | Read image, check dimensions                         |
| 3    | Client | Resize if >1920px (saves upload bandwidth)           |
| 4    | Client | Encode to JPEG, upload with progress bar             |
| 5    | Server | Decode bytes to numpy array, re-validate             |
| 6    | Server | YOLO plate detection → bounding boxes                |
| 7    | Server | Crop plate region with padding                       |
| 8    | Server | Preprocess (up to 7 enhancement variants)            |
| 9    | Server | PaddleOCR + EasyOCR with geometric filtering         |
| 10   | Server | Position-aware character correction                  |
| 11   | Server | Regex + RTO state code validation                    |
| 12   | Server | Majority voting across readings                      |
| 13   | Server | Return JSON result                                   |
| 14   | Client | Display result, optionally save annotated image      |

## Docker

### Build

```powershell
cd D:\LP_API
docker build -t anpr-api .
```

### Run

```powershell
# Foreground (see logs)
docker run -p 8000:8000 anpr-api

# Background
docker run -d -p 8000:8000 --name anpr anpr-api
```

### Manage

```powershell
docker ps                              # Check running containers
docker logs anpr                       # View logs
docker stop anpr                       # Stop
docker start anpr                      # Start again
docker stop anpr && docker rm anpr     # Remove container
```

### Rebuild after code changes

```powershell
docker stop anpr
docker rm anpr
docker build -t anpr-api .
docker run -d -p 8000:8000 --name anpr anpr-api
```

## Testing with Postman

1. Open Postman → New Request
2. **Health check:** GET `http://localhost:8000/health` → Send
3. **Recognize plate:**
   - POST `http://localhost:8000/recognize`
   - Body tab → form-data
   - Key: `file` (change type from Text to **File**)
   - Value: select your image
   - Click Send
4. **Validate plate:** GET `http://localhost:8000/validate?plate=KA01AB1234` → Send

## Known Issues

- **Full-scene 3840x2160 photos:** YOLO model struggles to detect small plates in large images. Pre-cropped plate images work much better. Fix: retrain YOLO with full-scene training data.
- **PaddleOCR version:** Must use 2.9.1. Version 3.x has breaking API changes and a Windows oneDNN bug.
- **First request is slow:** YOLO and OCR models warm up on the first inference. Subsequent requests are faster.

## Example Output

```text
Processing: img4.1.jpg
  Upload size: 6 KB (original: 6 KB)
  [██████████████████████████████] Done in 4.4s

  ────────────────────────────────────────────────────
  Source:     test_images\img4.1.jpg
  Plate:     KA03HH3511
  Confidence: 97.0%
  Stability:  100.0%
  Format:     VALID
  Detections: 1 plate(s) found
    [0] (19,15)→(183,90) conf=0.89
  Time:      4.4s
  ────────────────────────────────────────────────────
```
