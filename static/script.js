const constraints = {"video": {"facingMode": "environment"}}
let results = [];
const GLYPH_SIZE = 32;
const OVERLAY_MARGIN = 0.15;
const DEBUG=true;
const DATA_COLLECTION=true;

async function startCamera() {
    const userMedia = navigator.mediaDevices.getUserMedia(constraints);
    userMedia.then(stream => {
        const video = document.querySelector('.video-preview');
        video.srcObject = stream;
        setupOverlay();
    }).catch(err => {
        console.error("Error accessing camera: ", err);
    });
    document.querySelector('#start-camera').disabled = true;
    document.querySelector('#capture-ocr').style.display = "block";
}

async function setupOverlay() {
    const overlay = document.querySelector('.overlay');
    overlay.style.width = `${(1-2*OVERLAY_MARGIN)*100}%`;
    overlay.style.height = `${(1-2*OVERLAY_MARGIN)*100}%`;
    overlay.style.position = "absolute";
    overlay.style.top = `${OVERLAY_MARGIN*100}%`;
    overlay.style.left = `${OVERLAY_MARGIN*100}%`;
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
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const size = Math.min(video.videoWidth, video.videoHeight);
    const sx = (video.videoWidth - size) / 2;
    const sy = (video.videoHeight - size) / 2;
    canvas.width = size;
    canvas.height = size;
    ctx.drawImage(video, sx, sy, size, size, 0, 0, canvas.width, canvas.height);

    // First, split the image into 16 section (4 * 4 grid)
    const margin = OVERLAY_MARGIN * size;
    const mainImageSize = size - 2 * margin;
    const gridSize = 4;
    const sectionSize = mainImageSize / gridSize;
    const sectionData = [];
    for (let x = 0; x < gridSize; x++) {
        for (let y = 0; y < gridSize; y++) {
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
            if (DATA_COLLECTION && data && data.glyph) {

                const glyphCanvas = document.createElement('canvas');
                glyphCanvas.width = sectionSize;
                glyphCanvas.height = sectionSize;
                cv.imshow(glyphCanvas, data.glyph);
                sectionData.push({processed: glyphCanvas, raw: sectionCanvas});
            }
        }
    }
    if (DATA_COLLECTION && sectionData.length > 0) {
        sendImagePairs(sectionData);
    }
    //disable video
    video.pause();
    video.srcObject = null;
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
        {x: 0, y: 0},
        {x: roi.cols - 1, y: 0},
        {x: 0, y: roi.rows - 1},
        {x: roi.cols - 1, y: roi.rows - 1}
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
window.onload = function () {
    document.querySelector('#start-camera').onclick = startCamera;
    document.querySelector('#capture-ocr').onclick = captureOCR;
}