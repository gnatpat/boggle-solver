from glob import glob
import json
import os
from pathlib import Path
import re
from typing import Dict, List, Optional
import uuid
from fastapi import FastAPI, File, Request
from fastapi.concurrency import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
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


def save_labels(app: FastAPI):
    with open(LABELS_FILE, "w") as f:
        json.dump(app.state.labels, f, indent=2)


def refresh_files(app: FastAPI):
    # use glob to find all image files
    app.state.files = [
        f for f in glob(str(IMAGE_DIR / "*_raw.png"))
        if os.path.isfile(f)
    ]

@asynccontextmanager
async def lifespan(app):
    load_labels(app)
    refresh_files(app)
    yield
    save_labels(app)

app = FastAPI(lifespan=lifespan)

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

app.mount("/static", StaticFiles(directory="static"), name="static")