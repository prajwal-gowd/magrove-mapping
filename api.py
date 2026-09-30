import time
import io
import base64
import os
from functools import lru_cache
import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from config import SAVE_PATH, IMAGE_SIZE, DEVICE
from models.hybrid_model import HybridTCCFNet
from utils.inference import load_model_checkpoint, preprocess_image

app = FastAPI(title="Mangrove Classification API")

# ---------------------------------------------------------------------------
# CORS middleware for development
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure web directory exists for static files
os.makedirs("web", exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/tiff"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
TILE_SIZE = IMAGE_SIZE  # 224
TILE_STRIDE = 200       # 24 px overlap for smooth seams


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_model():
    """Instantiate HybridTCCFNet and load the best checkpoint (cached)."""
    model = HybridTCCFNet(pretrained_backbone=False)
    try:
        model = load_model_checkpoint(model, SAVE_PATH, DEVICE)
    except TypeError:
        # Fallback for older signature
        checkpoint = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    model = model.to(DEVICE)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def img_to_base64(img_arr: np.ndarray, mode: str = "L") -> str:
    """Encode a numpy array as a base64 PNG data-URL string."""
    img = Image.fromarray(img_arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def predict_tiles(image: Image.Image) -> np.ndarray:
    """
    Run tiled inference on an arbitrary-size RGB image.

    For images larger than 224×224 the image is split into overlapping
    224×224 tiles with a stride of 200.  Probability maps are averaged
    in overlap regions to produce smooth seams.

    Returns a float32 probability map of the same spatial size as
    the (possibly up-scaled) input.
    """
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    model = get_model()

    # Pad so tiles fit evenly
    if h > TILE_SIZE:
        pad_h = (TILE_STRIDE - (h - TILE_SIZE) % TILE_STRIDE) % TILE_STRIDE
    else:
        pad_h = TILE_SIZE - h
    if w > TILE_SIZE:
        pad_w = (TILE_STRIDE - (w - TILE_SIZE) % TILE_STRIDE) % TILE_STRIDE
    else:
        pad_w = TILE_SIZE - w

    img_padded = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    ph, pw = img_padded.shape[:2]

    prob_map = np.zeros((ph, pw), dtype=np.float32)
    count_map = np.zeros((ph, pw), dtype=np.float32)

    for y in range(0, ph - TILE_SIZE + 1, TILE_STRIDE):
        for x in range(0, pw - TILE_SIZE + 1, TILE_STRIDE):
            tile = Image.fromarray(img_padded[y : y + TILE_SIZE, x : x + TILE_SIZE])
            tensor = preprocess_image(tile)
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            tensor = tensor.to(DEVICE)

            with torch.no_grad():
                logits = model(tensor)
                probs = torch.sigmoid(logits).cpu().numpy().squeeze()

            prob_map[y : y + TILE_SIZE, x : x + TILE_SIZE] += probs
            count_map[y : y + TILE_SIZE, x : x + TILE_SIZE] += 1.0

    # Average overlapping regions and crop back to original size
    prob_map = prob_map / np.maximum(count_map, 1.0)
    return prob_map[:h, :w]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web application."""
    try:
        with open("web/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<html><body><h1>MangroveNet API is running</h1>"
            "<p><code>web/index.html</code> not found.</p></body></html>"
        )


@app.get("/api/health")
async def health():
    """Health-check endpoint with model metadata."""
    model_loaded = False
    error_msg = None
    try:
        get_model()
        model_loaded = True
    except Exception as exc:
        error_msg = str(exc)

    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "checkpoint_available": os.path.isfile(SAVE_PATH),
        "device": str(DEVICE),
        "model": "HybridTCCFNet (ResNet-50 + Transformer)",
        "image_size": IMAGE_SIZE,
        **({"error": error_msg} if error_msg else {}),
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept a satellite image upload, run mangrove segmentation, and return:
    - binary classification mask
    - confidence probability heatmap
    - green overlay on the original image
    - coverage statistics
    """
    start_time = time.time()

    # --- Validate upload ------------------------------------------------
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_format",
                "message": f"File type '{file.content_type}' is not supported. "
                f"Allowed: {', '.join(sorted(ALLOWED_TYPES))}",
            },
        )

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "file_too_large",
                "message": f"Upload exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit.",
            },
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_image", "message": str(exc)},
        )

    input_w, input_h = image.size

    # --- Ensure minimum dimensions for the model -----------------------
    min_dim = min(input_w, input_h)
    if min_dim < TILE_SIZE:
        scale = TILE_SIZE / min_dim
        new_w = max(TILE_SIZE, int(input_w * scale))
        new_h = max(TILE_SIZE, int(input_h * scale))
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    img_np = np.array(image)
    h, w = img_np.shape[:2]

    # --- Run tiled inference -------------------------------------------
    prob_map = predict_tiles(image)

    # Binary mask
    mask_np = (prob_map >= 0.65).astype(np.uint8) * 255
    mask_bool = mask_np > 0

    # --- Confidence heatmap (green gradient) ---------------------------
    conf_map = np.zeros((h, w, 3), dtype=np.uint8)
    # Dark-to-bright green proportional to probability
    conf_map[:, :, 0] = (prob_map * 34).astype(np.uint8)    # slight red tint
    conf_map[:, :, 1] = (prob_map * 255).astype(np.uint8)   # full green
    conf_map[:, :, 2] = (prob_map * 94).astype(np.uint8)    # slight blue tint

    # --- Overlay (50 % blend of green on mangrove pixels) --------------
    overlay = img_np.copy()
    green = np.array([34, 197, 94], dtype=np.float32)
    overlay[mask_bool] = np.clip(
        img_np[mask_bool].astype(np.float32) * 0.5 + green * 0.5,
        0, 255,
    ).astype(np.uint8)

    # --- Statistics ----------------------------------------------------
    total_pixels = h * w
    mangrove_pixels = int(np.sum(mask_bool))
    non_mangrove_pixels = total_pixels - mangrove_pixels
    coverage = round((mangrove_pixels / total_pixels) * 100, 2) if total_pixels else 0.0
    processing_time_ms = int((time.time() - start_time) * 1000)

    return {
        "mask": img_to_base64(mask_np, mode="L"),
        "overlay": img_to_base64(overlay, mode="RGB"),
        "confidence_map": img_to_base64(conf_map, mode="RGB"),
        "stats": {
            "mangrove_coverage_percent": coverage,
            "total_pixels": total_pixels,
            "mangrove_pixels": mangrove_pixels,
            "non_mangrove_pixels": non_mangrove_pixels,
            "input_width": input_w,
            "input_height": input_h,
            "output_width": w,
            "output_height": h,
            "processing_time_ms": processing_time_ms,
        },
    }


# ---------------------------------------------------------------------------
# Static file fallback (serves CSS / JS / images from web/)
# ---------------------------------------------------------------------------
app.mount("/web", StaticFiles(directory="web"), name="static-web")
