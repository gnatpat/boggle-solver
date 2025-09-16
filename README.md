# Boggle Solver

Live at https://natpat.net/boggle

A Web App for solving Boggle Boards. No server-side processing is required - everything runs **in browser**.

![Demo Screenshot](https://github.com/gnatpat/boggle-solver/blob/main/readme_assets/1000041431.png?raw=true)

## Overview
Technologies used:
- FastAPI
- PyTorch
- OpenCV
- OnnxRuntime

No front end framework - just vanilla HTML/CSS/JS.

## How does it work?

At a high level, the app does the following:

1. Takes an image of a Boggle board as input - a grid is overlaid on the image to help the user align it properly
2. The image is split into individual dice
3. Each die is pre-processed using OpenCV
4. The pre-processed images are passed through a CNN to recognize the letters on the dice
5. The detected letters are shown to the user for confirmation
6. The letters are passed to a word search algorithm that finds all valid words on the board
7. The found words are displayed to the user, sorted by length and score
8. The user can tap on words to see their paths on the board

### Letter Recognition Model
The letter recognition model is a small, custom CNN trained on a dataset of Boggle dice images. The dataset was created by taking photos of Boggle boards, splitting the images into individual dice, and labeling the letters on each die.

The model was trained using PyTorch and then converted to ONNX format for use in the web app with OnnxRuntime. The model achieves around 95% accuracy on a validation set.

## Local Development

This repository uses `uv` for local development.

### Serving the Web App

The Web App can be served by serving everything in the `static/` directory. For convenience, the development server (used for collecting training data and labelling) will also serve the static files. Run

```bash
uv run fastapi dev main.py --port 8000
```

And navigate to `http://localhost:8000/` to see the app.

#### Prerequisites

In order to run, the app requires:
1. A trained letter recognition model in ONNX format, named `boggle_cnn.onnx` in the `static/` directory. You can:
    - train your own using the code in `train_model.py` (see the Training section below) - contact me (find my email on e.g. my website) if you want the dataset used to train
    - download the model [here](http://natpat.net/boggle/boggle_cnn.onnx)
2. At least one trie file in `static/tries/`, along with a metadata file
    - these can be generated using the code in `make_word_lsit.py` and `generate_tries.py` - this also assumes you have a copy of [SCOWL](http://wordlist.aspell.net/) downloaded
    - or contact me if you want the tries I generated

### Training the Model

To train the letter recognition model, you will need:
 - a folder containing images of individual Boggle dice, pre-processed and named `<image_id>_processed.png` (e.g. `0001_processed.png`), at `images/`
 - a `labels.json` file mapping image ids to their corresponding letters, e.g. `{"0001": "A", "0002": "B", ...}`

You can then run

```bash
uv run model.py
```

to train the model. The trained model will be saved as `boggle_cnn.onnx`.

### Collecting Training Data

A server is provided to collect training data and help label it.

To run the server, use:

```bash
uv run fastapi dev main.py --port 8000
```

In `static/script.js`, set `DATA_COLLECTION` to `true` to enable data collection mode. This will store the raw and processed images of each die, along with a unique id, in the `images/` folder.

To label the collected data, navigate to `http://localhost:8000/static/labeller.html` on the running server. This page will show each collected image and allow you to enter the corresponding letter. The labels will be saved in `labels.json`.

