const constraints = { "video": { "facingMode": "environment" } }
const GLYPH_SIZE = 32;
const DEBUG = false;
const DATA_COLLECTION = false;
let predictions = [];
let trie = null;
let boardSolved = false;
let currentAnimation = null;


function getMargin() {
    const size = getSelectedSize();
    if (size === "4x4") {
        return 0.15;
    } else {
        return 0.125;
    }
}

async function startCamera() {
    const userMedia = navigator.mediaDevices.getUserMedia(constraints);
    userMedia.then(stream => {
        const video = document.querySelector('.video-preview');
        video.style.display = "block";
        video.srcObject = stream;
        setupOverlay();
        document.querySelector('#boggle-size').disabled = false;
    }).catch(err => {
        console.error("Error accessing camera: ", err);
    });
    document.querySelector('#start-camera').disabled = true;
    document.querySelector('#capture-ocr').style.display = "block";
    document.querySelector('.snapshot-canvas').style.display = "none";
    document.querySelector('#debug-info').innerHTML = '';
    document.querySelector('#boggle-board').innerHTML = '';
    document.querySelector('#results').innerHTML = '';
    document.querySelector('#solve-button-container').style.display = "none";
    document.querySelector('#boggle-board').style.display = "none";
    clearHighlightedWord();
    boardSolved = false;
    predictions = [];
}

function getSelectedSize() {
    const boggleSizeSelect = document.querySelector('#boggle-size');
    return boggleSizeSelect.value;
}

async function setupOverlay() {
    const selectedSize = getSelectedSize();
    let overlay;
    const overlay4x4 = document.querySelector('.overlay4x4');
    const overlay5x5 = document.querySelector('.overlay5x5');
    overlay4x4.style.display = "none";
    overlay5x5.style.display = "none";
    if (selectedSize === "4x4") {
        overlay = overlay4x4;
    } else {
        overlay = overlay5x5;
    }
    const margin = getMargin();
    overlay.style.display = "block";
    overlay.style.width = `${(1 - 2 * margin) * 100}%`;
    overlay.style.height = `${(1 - 2 * margin) * 100}%`;
    overlay.style.position = "absolute";
    overlay.style.top = `${margin * 100}%`;
    overlay.style.left = `${margin * 100}%`;
    overlay.style.pointerEvents = "none";
}

function addDebugData(sectionDiv, sectionCanvas, data) {
    sectionDiv.appendChild(sectionCanvas);
    const sectionSize = sectionCanvas.width;
    const contours = data.contours;
    const threshData = data.thresh;
    // Show thresh
    const threshCanvas = document.createElement('canvas');
    threshCanvas.width = sectionSize;
    threshCanvas.height = sectionSize;
    cv.imshow(threshCanvas, threshData);
    sectionDiv.appendChild(threshCanvas);

    // Show contours
    const contoursCanvas = document.createElement('canvas');
    contoursCanvas.width = sectionSize;
    contoursCanvas.height = sectionSize;
    sectionDiv.appendChild(contoursCanvas);
    // draw contours
    let dst = cv.Mat.zeros(threshData.size(), cv.CV_8UC3);
    for (let i = 0; i < contours.size(); ++i) {
        let color = new cv.Scalar(Math.round(Math.random() * 255), Math.round(Math.random() * 255),
            Math.round(Math.random() * 255));
        cv.drawContours(dst, contours, i, color, 1, cv.LINE_8);
    }
    cv.imshow(contoursCanvas, dst);

    if (data.maxContour) {
        // show max contour
        const maxContourCanvas = document.createElement('canvas');
        maxContourCanvas.width = sectionSize;
        maxContourCanvas.height = sectionSize;
        sectionDiv.appendChild(maxContourCanvas);
        // draw max contour
        let maxDst = cv.Mat.zeros(threshData.size(), cv.CV_8UC3);
        let maxContour = data.maxContour;
        let singleContourVector = new cv.MatVector();
        singleContourVector.push_back(maxContour);
        cv.drawContours(maxDst, singleContourVector, 0, new cv.Scalar(255, 0, 0), 1, cv.LINE_8);
        cv.imshow(maxContourCanvas, maxDst);
    }
    if (data.glyph) {
        const glyphCanvas = document.createElement('canvas');
        glyphCanvas.width = sectionSize;
        glyphCanvas.height = sectionSize;
        sectionDiv.appendChild(glyphCanvas);
        cv.imshow(glyphCanvas, data.glyph);
    }
}
async function captureOCR() {
    const video = document.querySelector('.video-preview');
    const canvas = document.querySelector('.snapshot-canvas');
    const ctx = canvas.getContext('2d');
    const size = Math.min(video.videoWidth, video.videoHeight);
    const sx = (video.videoWidth - size) / 2;
    const sy = (video.videoHeight - size) / 2;
    canvas.width = size;
    canvas.height = size;
    ctx.drawImage(video, sx, sy, size, size, 0, 0, canvas.width, canvas.height);

    // First, split the image into 16 section (4 * 4 grid)
    const margin = getMargin() * size;
    const mainImageSize = size - 2 * margin;
    const selectedSize = getSelectedSize();
    const gridSize = selectedSize === "4x4" ? 4 : 5;
    const sectionSize = mainImageSize / gridSize;
    const sectionData = [];
    for (let y = 0; y < gridSize; y++) {
        for (let x = 0; x < gridSize; x++) {
            const section = ctx.getImageData(x * sectionSize + margin, y * sectionSize + margin, sectionSize, sectionSize);
            // Preview each section in a canvas added to the page
            const sectionCanvas = document.createElement('canvas');
            sectionCanvas.width = sectionSize;
            sectionCanvas.height = sectionSize;
            const sectionCtx = sectionCanvas.getContext('2d');
            sectionCtx.putImageData(section, 0, 0);
            let data;
            try {
                data = preprocessDie(sectionCanvas);
            } catch (error) {
                console.error("Error processing section:", error);
            }
            if (data && DEBUG) {
                const sectionDiv = document.createElement('div');
                sectionDiv.classList.add('section');
                document.querySelector('#debug-info').appendChild(sectionDiv);
                addDebugData(sectionDiv, sectionCanvas, data);
            }
            if (data && data.glyph) {
                const glyphCanvas = document.createElement('canvas');
                glyphCanvas.width = sectionSize;
                glyphCanvas.height = sectionSize;
                cv.imshow(glyphCanvas, data.glyph);

                if (DATA_COLLECTION && data && data.glyph) {
                    sectionData.push({ processed: glyphCanvas, raw: sectionCanvas });
                }
                if (!inferenceSession) {
                    // wait for model to load for up to 5 seconds
                    const start = Date.now();
                    while (!inferenceSession && (Date.now() - start) < 5000) {
                        await new Promise(resolve => setTimeout(resolve, 100));
                    }
                    if (!inferenceSession) {
                        console.error("Model not loaded in time");
                        predictions.push(null);
                        continue;
                    }
                }
                predictions.push(await predictLetter(glyphCanvas));
            } else {
                predictions.push(null);
            }
        }
    }
    if (DATA_COLLECTION && sectionData.length > 0) {
        sendImagePairs(sectionData);
    }
    //disable video
    video.pause();
    video.srcObject = null;
    console.log(predictions);
    document.querySelector('#start-camera').disabled = false;
    document.querySelector('#capture-ocr').style.display = "none";
    document.querySelector('.snapshot-canvas').style.display = "block";
    document.querySelector('.video-preview').style.display = "none";
    document.querySelector('#boggle-size').disabled = true;
    document.querySelector('#solve-button-container').style.display = "block";
    document.querySelector('#solve-button').disabled = false;


    renderBoard();
    solveButton = document.querySelector('#solve-button');
    setSolveButtonState();
    solveButton.onclick = async () => {
        if (!trie) {
            console.error("Trie not loaded");
            return;
        }
        solveButton.disabled = true;
        solveBoard();
        document.querySelector('#solve-button-container').style.display = "none";
    }
    //scroll to board
    const board = document.querySelector('#boggle-board');
    board.scrollIntoView({ behavior: "smooth", block: "start", inline: "nearest" });
}

function setSolveButtonState() {
    const solveButton = document.querySelector('#solve-button');
    // check if predictions has any nulls
    if (predictions.length === 0 || predictions.includes(null)) {
        solveButton.disabled = true;
    } else {
        solveButton.disabled = false;
    }
}

async function predictLetter(canvas) {
    if (!inferenceSession) {
        console.error("Model not loaded");
        return;
    }
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data; // still RGBA, but R=G=B for grayscale

    // Convert to Float32Array, one channel, shape [1,1,32,32]
    const inputArray = new Float32Array(canvas.width * canvas.height);
    for (let i = 0; i < canvas.width * canvas.height; i++) {
        inputArray[i] = data[i * 4] / 255.0; // R channel only (G=B same)
    }

    // Wrap as ONNX tensor
    const inputTensor = new ort.Tensor('float32', inputArray, [1, 1, 32, 32]);

    const feeds = { input: inputTensor };
    const results = await inferenceSession.run(feeds);
    console.log(results);
    const outputArray = results.output.data;

    // 5. Find predicted letter
    const predictedIndex = outputArray.indexOf(Math.max(...outputArray));
    const predictedLetter = String.fromCharCode(65 + predictedIndex);

    console.log("Predicted letter:", predictedLetter);
    return predictedLetter;
}

const canvasToBlob = canvas =>
    new Promise(resolve => canvas.toBlob(resolve, "image/png"));

async function sendImagePairs(imagePairs) {
    const formData = new FormData();

    await Promise.all(
        imagePairs.map(async (pair, index) => {
            const rawBlob = await canvasToBlob(pair.raw);
            const processedBlob = await canvasToBlob(pair.processed);

            formData.append(`cube${index}_raw`, rawBlob, `cube${index}_raw.png`);
            formData.append(`cube${index}_processed`, processedBlob, `cube${index}_processed.png`);
        })
    );

    const response = await fetch("/upload-images", {
        method: "POST",
        body: formData,
    });

    const data = await response.json();
    console.log(data);
}



function preprocessDie(srcImg) {
    // Convert HTMLImage/canvas → cv.Mat
    let src = cv.imread(srcImg);
    let gray = new cv.Mat();
    let thresh = new cv.Mat();
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    const MARGIN = 0.2;

    // 1. Grayscale
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);

    cv.threshold(
        gray,   // input
        thresh,      // output
        0,           // ignored when using OTSU
        255,         // max value
        cv.THRESH_BINARY_INV + cv.THRESH_OTSU
    );

    let kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(2, 2));
    cv.morphologyEx(thresh, thresh, cv.MORPH_OPEN, kernel);

    // 4. Find contours
    cv.findContours(
        thresh,
        contours,
        hierarchy,
        cv.RETR_LIST,
        cv.CHAIN_APPROX_SIMPLE
    );

    // Pick largest contour (likely the letter)
    let maxArea = 0;
    let maxContour = null;
    for (let i = 0; i < contours.size(); i++) {
        const cnt = contours.get(i);
        const rect = cv.boundingRect(cnt);
        if (rect.width >= (1 - MARGIN * 2) * src.cols || rect.height >= (1 - MARGIN * 2) * src.rows) {
            continue;
        }
        const area = cv.contourArea(cnt);
        if (area > maxArea) {
            maxArea = area;
            maxContour = cnt;
        }
    }

    if (!maxContour) {
        // Cleanup
        return { thresh: thresh, contours: contours, maxContour: null, glyph: null };
    }

    //return {thresh: thresh, contours: contours, maxContour: maxContour};

    // Make square bounding box around maxContour, padding
    const rect = cv.boundingRect(maxContour);
    const size = Math.min(Math.max(rect.width, rect.height) * 1.5, thresh.cols);
    let x = Math.max(rect.x + rect.width / 2 - size / 2, 0);
    let y = Math.max(rect.y + rect.height / 2 - size / 2, 0);
    if (x + size > thresh.cols) {
        x = thresh.cols - size;
    }
    if (y + size > thresh.rows) {
        y = thresh.rows - size;
    }
    const roi = thresh.roi(new cv.Rect(x, y, size, size));


    // Resize to GLYPH_SIZE

    let mask = new cv.Mat.zeros(roi.rows + 2, roi.cols + 2, cv.CV_8UC1);
    let corners = [
        { x: 0, y: 0 },
        { x: roi.cols - 1, y: 0 },
        { x: 0, y: roi.rows - 1 },
        { x: roi.cols - 1, y: roi.rows - 1 }
    ];

    let newVal = new cv.Scalar(0);
    let loDiff = new cv.Scalar(50);
    let upDiff = new cv.Scalar(50);
    let flags = 4 | (255 << 8); // 4-connectivity

    corners.forEach(pt => {
        let pixelValue = roi.ucharAt(pt.y, pt.x);
        if (pixelValue !== 0) {
            console.log(roi);
            cv.floodFill(roi, mask, new cv.Point(pt.x, pt.y), newVal, new cv.Rect(), loDiff, upDiff, flags);
        }
    });

    let glyph = new cv.Mat();
    cv.resize(roi, glyph, new cv.Size(GLYPH_SIZE, GLYPH_SIZE), 0, 0, cv.INTER_AREA);
    // Cleanup
    src.delete(); gray.delete();
    roi.delete();

    return { thresh: thresh, contours: contours, maxContour: maxContour, glyph: glyph };

}

let inferenceSession = null;

async function loadModel() {
    inferenceSession = await ort.InferenceSession.create("boggle_cnn.onnx");
}

function updateOverlay() {
    setupOverlay();
}

function setTileText(tile, text, x, y, boardSize) {
    predictions[y * boardSize + x] = text;
    setSolveButtonState();
    if (text == 'Q') {
        text = 'Qu';
    }
    const tileText = tile.querySelector('.tile-text');
    tileText.textContent = text;
}

function renderBoard() {
    console.log("Rendering board with predictions:", predictions);
    const gridSize = getSelectedSize() === "4x4" ? 4 : 5;
    const board = document.querySelector('#boggle-board');
    board.innerHTML = ''; // Clear previous content
    board.style.display = "grid";
    board.style.gridTemplateColumns = `repeat(${gridSize}, 1fr)`;
    board.style.gridTemplateRows = `repeat(${gridSize}, 1fr)`;
    console.log(predictions.entries());
    for (let index = 0; index < predictions.length; index++) {
        const prediction = predictions[index];
        console.log(index, prediction);
        const x = index % gridSize;
        const y = Math.floor(index / gridSize);
        const tile = document.createElement('div');
        tile.className = 'boggle-tile';
        tile.id = `tile-${x}-${y}`;
        const tileText = document.createElement('span');
        tileText.className = 'tile-text';
        tileText.classList.add('highlight');
        tileText.id = `tile-text-${x}-${y}`;
        tile.appendChild(tileText);
        setTileText(tile, prediction, x, y, gridSize);
        board.appendChild(tile);
        tile.onclick = () => {
            clearHighlightedWord();
            tileText.textContent = '';
            const input = document.createElement('input');
            input.type = 'text';
            input.maxLength = 1;
            input.className = 'tile-input';
            input.onblur = () => {
                let val = input.value.toUpperCase();
                if (val.length === 1 && val >= 'A' && val <= 'Z') {
                    setTileText(tile, val, x, y, gridSize);
                    predictions[y * gridSize + x] = val;
                    if (boardSolved) {
                        document.querySelector('#results').innerHTML = '';
                        solveBoard();
                    }
                } else {
                    setTileText(tile, predictions[y * gridSize + x], x, y, gridSize);
                }
                console.log('removing');
                tile.removeChild(input);
            };
            input.onkeydown = (event) => {
                console.log(event);
                if (event.key === 'Enter' || event.key === 'Escape') {
                    input.blur();
                    event.preventDefault();
                }
            };
            input.onkeyup = (event) => {
                console.log(input.value);
                console.log(tile.textContent);
                let val = input.value.toUpperCase();
                if (val.length === 1 && val >= 'A' && val <= 'Z') {
                    input.blur();
                } else {
                    input.value = '';
                }
            }
            tile.appendChild(input);
            input.focus();
            if (!boardSolved) {
                input.scrollIntoView({ behavior: "smooth", block: "center", inline: "center" });
                // scroll into view after a short delay to allow for rendering
                setTimeout(() => {
                    input.scrollIntoView({ block: "center", inline: "center" });
                }, 100);
            }
        };
        console.log(x, y, prediction);
    }

    // Set font size based on cell width
    function updateFontSizes() {
        const cells = board.children;
        for (let i = 0; i < cells.length; i++) {
            const cellWidth = cells[i].offsetWidth;
            cells[i].style.fontSize = `${0.7 * cellWidth}px`; // 70% of cell width
        }
    }


    updateFontSizes();
    window.addEventListener('resize', updateFontSizes);
}

async function fetchAndDecompressJSON(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    const compressedData = await response.arrayBuffer();
    const compressedUint8Array = new Uint8Array(compressedData);
    const decompressedData = pako.ungzip(compressedUint8Array, { to: 'string' });
    return JSON.parse(decompressedData);
}

async function loadTrie() {
    const trieData = await fetchAndDecompressJSON('trie_data.json.gz');
    return trieData;
}

function getChild(node, char) {
    if (char === 'q') {
        if (node['q'] && node['q']['u']) {
            return node['q']['u'];
        } else {
            return null;
        }
    } else {
        return node[char];
    }
}

function findAllWords(board, trie, boardSize) {
    const directions = [
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1], [0, 1],
        [1, -1], [1, 0], [1, 1]
    ];

    const foundWords = [];

    function search(x, y, currentWord, node, visited, route) {
        const letter = board[y][x].toLowerCase();
        const childNode = getChild(node, letter);
        if (!childNode) {
            return;
        }
        visited.add(`${x},${y}`);
        route.push([x, y]);
        currentWord += letter;
        if (letter === 'q') {
            currentWord += 'u';
        }
        if ('$' in childNode) {
            foundWords.push({ word: currentWord, path: route.slice() });
        }
        for (const [dx, dy] of directions) {
            const newX = x + dx;
            const newY = y + dy;
            if (newX < 0 || newX >= boardSize || newY < 0 || newY >= boardSize) {
                continue;
            }
            if (visited.has(`${newX},${newY}`)) {
                continue
            }
            search(newX, newY, currentWord, childNode, visited, route);
        }
        visited.delete(`${x},${y}`);
        route.pop();
    }

    for (let i = 0; i < boardSize; i++) {
        for (let j = 0; j < boardSize; j++) {
            search(i, j, '', trie, new Set(), []);
        }
    }
    // deduplicate foundWords
    const uniqueWords = {};
    for (const entry of foundWords) {
        if (!(entry.word in uniqueWords)) {
            uniqueWords[entry.word] = entry.path;
        }
    }
    return uniqueWords;
}

function highlightPath(path) {
    if (!path) return;
    // clear previous highlights
    document.querySelectorAll('.tile-text').forEach(tile => {
        tile.classList.remove('highlight');
    });
    path.forEach(([x, y]) => {
        const tile = document.querySelector(`#tile-text-${x}-${y}`);
        if (tile) {
            tile.classList.add('highlight');
        }
    });
    // draw path on canvas
    const canvas = document.querySelector('#path-canvas');
    const board = document.querySelector('#boggle-board');
    canvas.width = board.clientWidth;
    canvas.height = board.clientHeight;
    canvas.style.width = `${board.clientWidth}px`;
    canvas.style.height = `${board.clientHeight}px`;
    canvas.style.position = 'absolute';
    canvas.style.top = `${board.offsetTop}px`;
    canvas.style.left = `${board.offsetLeft}px`;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const color = 'rgba(53, 117, 253, 1)';
    const radius = 8;
    const lastIndex = path.length - 1;
    ctx.strokeStyle = color;
    ctx.lineWidth = 7;
    ctx.lineCap = 'round';
    path.forEach(([x, y], index) => {
        const center = getTileCenter(x, y);
        if (center) {
            const [centerX, centerY] = center;
            if (index === 0) {
                // draw filled in circle at start
                ctx.beginPath();
                ctx.moveTo(centerX + radius, centerY);
                ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
                ctx.fillStyle = color;
                ctx.fill();
                ctx.stroke();

                ctx.beginPath();
                // move to center
                ctx.moveTo(centerX, centerY);
            } else {
                ctx.lineTo(centerX, centerY);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(centerX, centerY);
            }
            if (index === lastIndex) {
                ctx.stroke();
                // draw arrowhead
                const [prevX, prevY] = getTileCenter(path[index - 1][0], path[index - 1][1]);
                const angle = Math.atan2(centerY - prevY, centerX - prevX);
                const arrowLength = 15;
                const arrowAngle = Math.PI / 4; // 45 degrees
                ctx.beginPath();
                ctx.moveTo(centerX, centerY);
                ctx.lineTo(centerX - arrowLength * Math.cos(angle - arrowAngle), centerY - arrowLength * Math.sin(angle - arrowAngle));
                ctx.moveTo(centerX, centerY);
                ctx.lineTo(centerX - arrowLength * Math.cos(angle + arrowAngle), centerY - arrowLength * Math.sin(angle + arrowAngle));
                ctx.stroke();
            }
        }
    });
    ctx.stroke();

}

function getTile(x, y) {
    return document.querySelector(`#tile-${x}-${y}`);
}

function getTileTextTag(x, y) {
    return document.querySelector(`#tile-text-${x}-${y}`);
}

function getTileCenter(x, y) {
    const tile = getTile(x, y);
    const board = document.querySelector('#boggle-board');
    if (tile && board) {
        const rect = tile.getBoundingClientRect();
        const boardRect = board.getBoundingClientRect();
        const centerX = rect.left - boardRect.left + rect.width / 2;
        const centerY = rect.top - boardRect.top + rect.height / 2;
        return [centerX, centerY];
    }
    return null;
}

function hidePathCanvas() {
    const canvas = document.querySelector('#path-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.style.width = '0px';
    canvas.style.height = '0px';
}

function solveBoard() {
    const gridSize = getSelectedSize() === "4x4" ? 4 : 5;
    if (predictions.length !== gridSize * gridSize) {
        console.error("Predictions length does not match grid size");
        return;
    }
    const board = [];
    for (let i = 0; i < gridSize; i++) {
        board.push(predictions.slice(i * gridSize, (i + 1) * gridSize));
    }
    console.log(board);
    const foundWords = findAllWords(board, trie, gridSize);
    console.log("Found words:", foundWords);
    // group by length
    const grouped = {};
    Object.entries(foundWords).forEach(([word, path]) => {
        const len = word.length;
        if (!grouped[len]) {
            grouped[len] = [];
        }
        grouped[len].push({ word, path });
    });
    // sort each group alphabetically
    for (const len in grouped) {
        grouped[len].sort((a, b) => a.word.localeCompare(b.word));
    }
    console.log("Grouped words:", grouped);
    const sizes = Object.keys(grouped).sort((a, b) => b - a);
    const resultsDiv = document.querySelector('#results');
    resultsDiv.innerHTML = '';
    const template = document.querySelector('#word-length-section');
    const wordEntryTemplate = document.querySelector('#word-entry');
    sizes.forEach(len => {
        const section = template.content.cloneNode(true);
        section.querySelector('.word-length').textContent = `${len} letter words (${grouped[len].length})`;
        const ul = section.querySelector('.word-list');
        grouped[len].forEach(entry => {
            const li = wordEntryTemplate.content.cloneNode(true);
            const wordSpan = li.querySelector('.word');
            wordSpan.textContent = entry.word;
            wordSpan.onclick = () => {
                highlightWord(entry);
            };
            ul.appendChild(li);
        });
        resultsDiv.appendChild(section);
    });
    if (!boardSolved) {
        const boardElement = document.querySelector('#boggle-board');
        const topOfBoard = boardElement.getBoundingClientRect().top + window.scrollY;
        window.scrollTo({ top: topOfBoard, behavior: 'smooth' });
    }
    boardSolved = true;
    // scroll results into view
}

function highlightWord(entry) {
    const path = entry.path;
    const word = entry.word;
    highlightPath(path);
    startAnimation(path);
    document.querySelector('#current-path-info').style.display = 'flex';
    const wordLength = word.length;
    document.querySelector('#highlighted-word').textContent = `${word} (${wordLength})`;
}

function clearHighlightedWord() {
    // highlight all tiles
    document.querySelectorAll('.tile-text').forEach(tile => {
        tile.classList.add('highlight');
    });
    hidePathCanvas();
    if (currentAnimation) {
        currentAnimation.stopped = true;
    }
    document.querySelector('#current-path-info').style.display = 'none';
}

// Simple sleep helper
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function animateWordPath(path, controller, delay = 200, pause = 1000) {
    if (!path || path.length === 0) return;

    await sleep(300); // initial delay before starting

    while (!controller.stopped) {
        // Highlight tiles in order
        for (const [x, y] of path) {
            if (controller.stopped) break;
            const tile = getTileTextTag(x, y);
            tile.classList.add('animation-highlight');
            // remove after slightly longer delay
            window.setTimeout(() => {
                tile.classList.remove('animation-highlight');
            }, 250);
            await sleep(delay);
        }

        if (controller.stopped) break;

        // Keep highlighted for a pause
        await sleep(pause);
        if (controller.stopped) break;
    }

    // Ensure tiles are reset at the end
    for (const [x, y] of path) {
        const tile = getTileTextTag(x, y);
        tile.classList.remove('animation-highlight');
    }
}

function startAnimation(path) {
    // Stop previous animation if any
    if (currentAnimation) {
        currentAnimation.stopped = true;
    }

    // Create new controller for this animation
    const controller = { stopped: false };
    currentAnimation = controller;

    // Start animation
    animateWordPath(path, controller);
}

let startTime = null;
let timerId = null;
const duration = 180;
let pausedTime = null;
let wakeLock = null;
let shouldHaveWakeLock = false;

async function requestWakeLock() {
    if ('wakeLock' in navigator) {
        try {
            wakeLock = await navigator.wakeLock.request('screen');
            shouldHaveWakeLock = true;
            wakeLock.addEventListener('release', () => {
                console.log('Wake Lock was released');
                wakeLock = null;
            });
            console.log('Wake Lock is active');
        } catch (err) {
            console.error(`${err.name}, ${err.message}`);
        }
    } else {
        console.warn('Wake Lock API not supported');
    }
}

async function releaseWakeLock() {
    if (wakeLock) {
        try {
            await wakeLock.release();
            wakeLock = null;
            shouldHaveWakeLock = false;
            console.log('Wake Lock released');
        } catch (err) {
            console.error('Error releasing Wake Lock:', err);
        }
    }
}

function startTimer(startFromPaused = false) {
    if (startFromPaused) {
        if (!pausedTime) return;
        const pausedDuration = performance.now() - pausedTime;
        startTime += pausedDuration;
        pausedTime = null;
    } else {
        startTime = performance.now();
        pausedTime = null;
    }
    if (timerId) {
        clearInterval(timerId);
    }
    requestWakeLock();
    updateTimer(); // initial call to set immediately
    timerId = setInterval(updateTimer, 100);
    document.querySelector('#reset-timer').style.display = 'inline-block';
    document.querySelector('#pause-timer').style.display = 'inline-block';
    document.querySelector('#start-timer').style.display = 'none';
    document.querySelector('#resume-timer').style.display = 'none';
}

function getDisplayTime(time) {
    const seconds = Math.ceil(time / 1000);
    const minutes = Math.floor(seconds / 60);
    const displaySeconds = seconds % 60;
    return `${minutes}:${displaySeconds.toString().padStart(2, '0')}`;
}

function updateTimer() {
    if (!startTime) return;
    const elapsed = performance.now() - startTime;
    const remaining = Math.max(duration * 1000 - elapsed, 0);
    const displayTime = getDisplayTime(remaining);
    const timeRemainingElement = document.querySelector('#time-remaining');
    const currentDisplay = timeRemainingElement.textContent;
    if (currentDisplay !== displayTime) {
        timeRemainingElement.textContent = displayTime;
    }

    if (remaining <= 0) {
        clearInterval(timerId);
        document.querySelector('#time-remaining').textContent = "Time's up!";
        document.querySelector('#pause-timer').style.display = 'none';
        document.querySelector('#resume-timer').style.display = 'none';
        timerId = null;
        pausedTime = null;
        document.querySelector('#timer-sound').play();
    }
}

function pauseTimer() {
    if (timerId) {
        clearInterval(timerId);
        timerId = null;
        pausedTime = performance.now();
    }
    releaseWakeLock();
    document.querySelector('#pause-timer').style.display = 'none';
    document.querySelector('#resume-timer').style.display = 'inline-block';
}

function resetTimer() {
    pauseTimer();
    startTime = null;
    document.querySelector('#time-remaining').textContent = getDisplayTime(duration * 1000);
    document.querySelector('#reset-timer').style.display = 'none';
    document.querySelector('#pause-timer').style.display = 'none';
    document.querySelector('#resume-timer').style.display = 'none';
    document.querySelector('#start-timer').style.display = 'inline-block';
    pausedTime = null;
    const sound = document.querySelector('#timer-sound');
    sound.currentTime = 0;
    sound.pause();
}

function resumeTimer() {
    if (pausedTime) {
        startTimer(true);
    }
}

function visibilityChange() {
    if (document.visibilityState === 'visible') {
        if (shouldHaveWakeLock && !wakeLock) {
            requestWakeLock();
        }
    } else {
        if (wakeLock) {
            releaseWakeLock();
        }
    }
}

document.addEventListener('visibilitychange', visibilityChange);

window.onload = async function () {
    document.querySelector('#start-camera').onclick = startCamera;
    document.querySelector('#capture-ocr').onclick = captureOCR;
    document.querySelector('#boggle-size').onchange = updateOverlay;
    document.querySelector('#clear-path').onclick = clearHighlightedWord;
    document.querySelector('#start-timer').onclick = function() {
        startTimer();
    }
    document.querySelector('#pause-timer').onclick = pauseTimer;
    document.querySelector('#reset-timer').onclick = resetTimer;
    document.querySelector('#resume-timer').onclick = resumeTimer;
    resetTimer();
    await loadModel();
    trie = await loadTrie();
}