"""Single-service website and inference API: uvicorn app:app."""
from pathlib import Path
from threading import Lock

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from network import NeuralNetwork
from params import load_params

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Digit Lab")
model = NeuralNetwork([784, 16, 16, 10])
load_params(model, ROOT / "params.npz")
model_lock = Lock()
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class Drawing(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pixels: list[float] = Field(min_length=784, max_length=784)


def prepare_pixels(pixels):
    """Fit ink to MNIST's 20-pixel box and center its mass in 28x28."""
    ink = np.asarray(pixels, dtype=float).reshape(28, 28)
    if not np.isfinite(ink).all() or (ink < 0).any() or (ink > 1).any():
        raise ValueError("Pixels must be finite numbers between 0 and 1.")
    ys, xs = np.nonzero(ink > 0)
    if not len(xs):
        raise ValueError("Draw a digit first.")
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = 20 / max(crop.shape)
    size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
    resized = np.asarray(Image.fromarray((crop * 255).astype('uint8')).resize(
        size, Image.Resampling.LANCZOS), dtype=float) / 255
    result = np.zeros((28, 28))
    y, x = (28 - size[1]) // 2, (28 - size[0]) // 2
    result[y:y + size[1], x:x + size[0]] = resized
    yy, xx = np.indices(result.shape)
    dy = round(13.5 - (yy * result).sum() / result.sum())
    dx = round(13.5 - (xx * result).sum() / result.sum())
    shifted = np.zeros_like(result)
    for row, col in zip(*np.nonzero(result)):
        if 0 <= row + dy < 28 and 0 <= col + dx < 28:
            shifted[row + dy, col + dx] = result[row, col]
    return shifted.flatten()


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
def predict(drawing: Drawing):
    try:
        pixels = prepare_pixels(drawing.pixels)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    # The original forward pass stores layer state, so serialize inference.
    with model_lock:
        scores = model.forward(pixels).copy()
    return {"digit": int(np.argmax(scores)), "scores": scores.tolist(),
            "processed_pixels": pixels.tolist()}
