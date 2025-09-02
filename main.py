from glob import glob
import json
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
import uuid
from fastapi import BackgroundTasks, FastAPI, File, Request
from fastapi.concurrency import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from starlette.datastructures import UploadFile

from attrs import frozen, define

IMAGE_DIR = Path("images")
IMAGE_DIR.mkdir(exist_ok=True)
LABELS_FILE = "labels.json"

def load_labels(app: FastAPI):
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            app.state.labels = json.load(f)
    else:
        app.state.labels = {}


def save_labels(labels: Dict[str, str]):
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f, indent=2)


def refresh_files(app: FastAPI):
    # use glob to find all image files
    app.state.image_ids = []
    filepaths = glob(str(IMAGE_DIR / "*_raw.png"))
    for f in filepaths:
        if not os.path.isfile(f):
            continue
        basepath = os.path.basename(f)
        image_id = get_image_id(basepath)
        if image_id:
            app.state.image_ids.append(image_id)
    print(app.state.image_ids)

def get_image_id(filename: str) -> Optional[str]:
    match = re.match(r"(\w+)_raw\.png", filename)
    if match:
        return match.group(1)
    return None

def get_image_filepaths(image_id: str) -> Tuple[str, str]:
    raw_path = IMAGE_DIR / f"{image_id}_raw.png"
    processed_path = IMAGE_DIR / f"{image_id}_processed.png"
    return '/' + str(raw_path), '/' + str(processed_path)


@asynccontextmanager
async def lifespan(app):
    load_labels(app)
    refresh_files(app)
    yield
    save_labels(app.state.labels)

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index():
    # serve index.html
    return HTMLResponse(open("static/index.html").read())

@define
class ImagePair:
    raw: Optional[UploadFile]
    processed: Optional[UploadFile]

@app.post("/upload-images")
async def upload_images(request: Request):
    form = await request.form()
    pairs: Dict[str, ImagePair] = {}

    print(form)
    pattern = re.compile(r"cube(\d+)_(raw|processed)")
    for key, value in form.multi_items():
        if not isinstance(value, UploadFile):
            continue
        match = pattern.match(key)
        if not match:
            continue
        cube_id = match.group(1)
        image_type = match.group(2)
        pairs.setdefault(cube_id, ImagePair(raw=None, processed=None))
        if image_type == "raw":
            pairs[cube_id].raw = value
        else:
            pairs[cube_id].processed = value

    for cube_id, pair in pairs.items():
        new_id = uuid.uuid4().hex
        if not pair.raw or not pair.processed:
            continue
        raw_filename = f"{new_id}_raw.png"
        processed_filename = f"{new_id}_processed.png"
        # write images to upload dir
        raw_contents = await pair.raw.read()
        processed_contents = await pair.processed.read()
        with open(IMAGE_DIR / raw_filename, "wb") as f:
            f.write(raw_contents)
        with open(IMAGE_DIR / processed_filename, "wb") as f:
            f.write(processed_contents)


@app.get("/refresh")
async def refresh():
    refresh_files(app)
    return {"count": len(app.state.image_ids)}

@app.get("/progress")
async def progress():
    total = len(app.state.image_ids)
    labeled = len(app.state.labels)
    return {
        "labeled": labeled,
        "total": total,
        "remaining": total - labeled if total else 0,
        "done": total > 0 and labeled >= total,
    }

@app.get("/next")
async def next_image():
    for image_id in app.state.image_ids:
        if image_id not in app.state.labels:
            raw_path, processed_path = get_image_filepaths(image_id)
            return {"image_id": image_id, "raw_path": raw_path, "processed_path": processed_path}
    return {"done": True}

@app.get("/image_id/{image_id}")
async def get_image_details(image_id: str):
    if image_id in app.state.labels:
        raw_path, processed_path = get_image_filepaths(image_id)
        return {"image_id": image_id, "label": app.state.labels[image_id], "raw_path": raw_path, "processed_path": processed_path}
    return {"error": "Image not found"}, 404

@app.get("/images/{filename}")
async def get_image(filename: str):
    path = os.path.join(IMAGE_DIR, filename)
    return FileResponse(path)

@app.post("/label/{image_id}/{label}")
async def label_image(image_id: str, label: str, background_tasks: BackgroundTasks):
    if image_id not in app.state.image_ids:
        return {"error": "Image not found"}, 404
    # ensure label is A-Z and ? in uppercase
    label = label.upper()
    if not re.match(r"^[A-Z?]$", label):
        return {"error": "Invalid label"}, 400
    app.state.labels[image_id] = label
    # save labels as background task
    background_tasks.add_task(save_labels, app.state.labels)
    return {"status": "success"}