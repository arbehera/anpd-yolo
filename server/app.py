"""
ANPR FastAPI Server
Uses the ORIGINAL pipeline code — no modular rewrite.

Run:  python run.py
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import asyncio
import logging
import uuid
import re
from contextlib import asynccontextmanager
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .anpr_pipeline import IndianANPR

logger = logging.getLogger("anpr.api")

# --- Config -----------------------------------------------------------------
MAX_UPLOAD_MB = 10
MAX_IMAGE_DIMENSION = 4000
MIN_IMAGE_DIMENSION = 50
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
INFERENCE_TIMEOUT = 60  # seconds


# --- Response models --------------------------------------------------------

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class PlateResult(BaseModel):
    success: bool
    text: str
    confidence: float
    stability: float
    valid_format: bool
    detections: List[BoundingBox] = Field(default_factory=list)


class DetectionResponse(BaseModel):
    count: int
    detections: List[BoundingBox]


class ValidationResponse(BaseModel):
    input: str
    normalized: str
    valid_format: bool
    matched_pattern: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    paddle_available: bool
    easyocr_available: bool
    state_codes_loaded: int


# --- App lifespan -----------------------------------------------------------

anpr: Optional[IndianANPR] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global anpr
    logger.info("Loading ANPR pipeline...")
    anpr = IndianANPR(
        yolo_model_path="lp_model.pt",
        debug_dir="debug_output",
        save_debug=True,
        ocr_engine="auto",  
    )
    logger.info("ANPR pipeline ready.")
    yield
    anpr = None


app = FastAPI(
    title="Indian ANPR API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


# --- Middleware -------------------------------------------------------------

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", "unknown")
    logger.exception(f"[{rid}] Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"error": str(exc)})


# --- Helpers ----------------------------------------------------------------

def get_anpr() -> IndianANPR:
    if anpr is None:
        raise HTTPException(503, "ANPR pipeline not initialized")
    return anpr


async def validate_and_decode(file: UploadFile) -> np.ndarray:
    ct = file.content_type or ""
    if ct not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(415, f"Unsupported type: {ct}")

    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_MB} MB)")

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")

    h, w = img.shape[:2]
    if max(h, w) > MAX_IMAGE_DIMENSION:
        raise HTTPException(413, f"Image too large ({w}x{h}px)")
    if min(h, w) < MIN_IMAGE_DIMENSION:
        raise HTTPException(400, f"Image too small ({w}x{h}px)")

    return img


# --- Endpoints --------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    return {"service": "Indian ANPR API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(pipeline: IndianANPR = Depends(get_anpr)):
    return HealthResponse(
        status="ok",
        paddle_available=pipeline.paddle_reader is not None,
        easyocr_available=pipeline.easy_reader is not None,
        state_codes_loaded=len(pipeline.state_codes),
    )


@app.post("/recognize", response_model=PlateResult, tags=["anpr"])
async def recognize(
    file: UploadFile = File(...),
    pipeline: IndianANPR = Depends(get_anpr),
):
    """Full pipeline: detect → enhance → OCR → correct → validate → vote."""
    img = await validate_and_decode(file)
    try:
        result = await asyncio.wait_for(
            run_in_threadpool(pipeline.process_array, img),
            timeout=INFERENCE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, "Inference timed out")
    return PlateResult(**result)


@app.post("/detect", response_model=DetectionResponse, tags=["anpr"])
async def detect(
    file: UploadFile = File(...),
    pipeline: IndianANPR = Depends(get_anpr),
):
    """Detection only — bounding boxes, no OCR."""
    img = await validate_and_decode(file)
    dets = await run_in_threadpool(pipeline.detect_plate, img)
    boxes = [BoundingBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3], confidence=round(c, 3))
             for b, c in dets]
    return DetectionResponse(count=len(boxes), detections=boxes)


@app.get("/validate", response_model=ValidationResponse, tags=["anpr"])
def validate(
    plate: str = Query(..., description="e.g. KA01AB1234"),
    pipeline: IndianANPR = Depends(get_anpr),
):
    """Validate a plate string. No image needed."""
    normalized = re.sub(r'[^A-Z0-9]', '', plate.upper().strip())
    matched = None
    for name, pattern in pipeline.patterns.items():
        if pattern.match(normalized):
            matched = name
            break
    return ValidationResponse(
        input=plate,
        normalized=normalized,
        valid_format=pipeline._is_valid(normalized),
        matched_pattern=matched,
    )


@app.get("/state-codes", tags=["reference"])
def state_codes(pipeline: IndianANPR = Depends(get_anpr)):
    codes = sorted(pipeline.state_codes)
    return {"count": len(codes), "codes": codes}