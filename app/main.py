import os
import pickle
import numpy as np
import cv2
from typing import Dict
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from PIL import Image

from app.feature_extractor import extract_image_features
from app.search import top_k_similar

app = FastAPI(title="Visual Search API", version="1.0")

DATA_DIR = "data"
IMAGE_DB_PATH = os.path.join(DATA_DIR, "product_image_features.pkl")
PRODUCT_DATA_PATH = os.path.join(DATA_DIR, "product_data.pkl")

APP_API_KEY = os.getenv("APP_API_KEY", "123456")  # change later in hosting env

image_db: Dict[str, np.ndarray] = {}
product_data = None  # dataframe (optional)

@app.on_event("startup")
def load_db():
    global image_db, product_data

    if not os.path.exists(IMAGE_DB_PATH):
        raise RuntimeError(f"Missing {IMAGE_DB_PATH}")

    with open(IMAGE_DB_PATH, "rb") as f:
        image_db = pickle.load(f)

    # Optional: load product metadata for nicer responses
    if os.path.exists(PRODUCT_DATA_PATH):
        with open(PRODUCT_DATA_PATH, "rb") as f:
            product_data = pickle.load(f)

@app.get("/health")
def health():
    return {"status": "ok", "image_db_products": len(image_db)}

# def require_key(x_api_key: str | None):
#     if x_api_key != APP_API_KEY:
#         raise HTTPException(status_code=401, detail="Invalid API key")

def require_key(x_api_key: str | None, x_rapidapi_key: str | None):
    key = x_api_key or x_rapidapi_key
    if key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def enrich(product_link: str):
    # If product_data exists, return details for this product_link
    if product_data is None:
        return {"product_link": product_link}

    row = product_data[product_data["Product Link"] == product_link]
    if row.empty:
        return {"product_link": product_link}

    r = row.iloc[0]
    return {
        "product_link": product_link,
        "title": r.get("Title"),
        "category": r.get("Category"),
        "price": r.get("Price"),
        "color": r.get("Color"),
        "pictures": r.get("Pictures"),
    }

@app.post("/search/image")
# async def search_image(
#     file: UploadFile = File(...),
#     top_k: int = 5,
#     x_api_key: str | None = Header(default=None, alias="X-API-Key"),
# ):
#     require_key(x_api_key)

async def search_image(
    file: UploadFile = File(...),
    top_k: int = 5,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_rapidapi_key: str | None = Header(default=None, alias="X-RapidAPI-Key"),
):
    require_key(x_api_key, x_rapidapi_key)


    content = await file.read()
    img = Image.open(BytesIO(content))
    qvec = extract_image_features(img)

    matches = top_k_similar(qvec, image_db, k=top_k)
    return {"results": [{**enrich(p), "score": s} for p, s in matches]}

def extract_frames(video_bytes: bytes, frame_interval: int = 10):
    tmp = "tmp_video.mp4"
    with open(tmp, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(tmp)
    frames = []
    count = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if count % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        count += 1

    cap.release()
    try:
        os.remove(tmp)
    except:
        pass

    return frames

@app.post("/search/video")
# async def search_video(
#     file: UploadFile = File(...),
#     top_k: int = 5,
#     frame_interval: int = 10,
#     x_api_key: str | None = Header(default=None, alias="X-API-Key"),
# ):
#     require_key(x_api_key)

async def search_video(
    file: UploadFile = File(...),
    top_k: int = 5,
    frame_interval: int = 10,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_rapidapi_key: str | None = Header(default=None, alias="X-RapidAPI-Key"),
):
    require_key(x_api_key, x_rapidapi_key)


    content = await file.read()
    frames = extract_frames(content, frame_interval=frame_interval)

    if not frames:
        raise HTTPException(status_code=400, detail="Could not extract frames")

    vecs = [extract_image_features(fr) for fr in frames]
    qvec = np.mean(vecs, axis=0)

    matches = top_k_similar(qvec, image_db, k=top_k)
    return {
        "frames_used": len(frames),
        "results": [{**enrich(p), "score": s} for p, s in matches]
    }
