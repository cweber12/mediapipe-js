// pose_landmarker_frame.js
// Pose detection on extracted frames

// Import MediaPipe Tasks Vision bundle
import {
    FilesetResolver,
    PoseLandmarker, 
    DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/vision_bundle.js";
import { VideoFrameExtractor } from './video_frame_extractor.js';

// Run pose detection on extracted video frames at specified interval
export async function runPoseDetectionOnFrames(
    videoEl, canvasEl, statusEl, poseResults, intervalSeconds, frameNav, frameCounter, cropRect
) {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );
    const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            delegate: "GPU"
        },
        runningMode: "IMAGE",
        numPoses: 1
    });

    poseResults.length = 0;
    let crop = cropRect ? { ...cropRect } : null;
    let frameIdx = 0;

    const extractor = new VideoFrameExtractor(videoEl, canvasEl);

    await extractor.extractFrames(intervalSeconds, async (frameUrl, t, frameWidth, frameHeight) => {
        // Crop logic
        const img = new Image();
        img.src = frameUrl;
        await new Promise(resolve => { img.onload = resolve; });

        let croppedCanvas = document.createElement('canvas');
        let croppedCtx = croppedCanvas.getContext('2d');
        if (crop) {
            croppedCanvas.width = crop.width;
            croppedCanvas.height = crop.height;
            croppedCtx.drawImage(
                img,
                crop.left, crop.top, crop.width, crop.height,
                0, 0, crop.width, crop.height
            );
        } else {
            croppedCanvas.width = img.width;
            croppedCanvas.height = img.height;
            croppedCtx.drawImage(img, 0, 0, img.width, img.height);
        }

        // Run pose detection on cropped image
        const croppedImg = new Image();
        croppedImg.src = croppedCanvas.toDataURL();
        await new Promise(resolve => { croppedImg.onload = resolve; });

        const result = poseLandmarker.detect(croppedImg);

        // Center crop on hip midpoint for next frame
        if (crop && result.landmarks && result.landmarks.length > 0) {
            const landmarkSet = result.landmarks[0];
            const leftHipIdx = 23;
            const rightHipIdx = 24;
            if (landmarkSet[leftHipIdx] && landmarkSet[rightHipIdx]) {
                const centerX_cropped = (landmarkSet[leftHipIdx].x + landmarkSet[rightHipIdx].x) / 2;
                const centerY_cropped = (landmarkSet[leftHipIdx].y + landmarkSet[rightHipIdx].y) / 2;
                const centerX_original = crop.left + centerX_cropped * crop.width;
                const centerY_original = crop.top + centerY_cropped * crop.height;
                crop.left = Math.max(0, Math.round(centerX_original - crop.width / 2));
                crop.top = Math.max(0, Math.round(centerY_original - crop.height / 2));
                crop.left = Math.min(crop.left, frameWidth - crop.width);
                crop.top = Math.min(crop.top, frameHeight - crop.height);
            }
        }

        // Draw pose landmarks on original frame (with offset)
        canvasEl.width = frameWidth;
        canvasEl.height = frameHeight;
        const ctx = canvasEl.getContext('2d');
        ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
        ctx.drawImage(img, 0, 0, frameWidth, frameHeight);

        let offsetLandmarks = [];
        if (result.landmarks && result.landmarks.length > 0) {
            for (const landmarkSet of result.landmarks) {
                offsetLandmarks = landmarkSet.map(lm => ({
                    ...lm,
                    x: lm.x * crop.width + crop.left,
                    y: lm.y * crop.height + crop.top,
                    z: lm.z,
                    visibility: lm.visibility
                }));
                // Draw circles
                offsetLandmarks.forEach(lm => {
                    ctx.beginPath();
                    ctx.arc(lm.x, lm.y, 4, 0, 2 * Math.PI);
                    ctx.fillStyle = 'lime';
                    ctx.fill();
                });
                // Draw connectors
                const drawingUtils = new DrawingUtils(ctx);
                const normalizedLandmarks = offsetLandmarks.map(lm => ({
                    x: lm.x / canvasEl.width,
                    y: lm.y / canvasEl.height,
                    z: lm.z,
                    visibility: lm.visibility
                }));
                drawingUtils.drawConnectors(
                    normalizedLandmarks,
                    PoseLandmarker.POSE_CONNECTIONS,
                    { color: 'lime', lineWidth: 2 }
                );
            }
        }

        // Save pose data and preview image
        poseResults.push({
            frameIdx: frameIdx++,
            frameUrl: canvasEl.toDataURL(),
            landmarks: offsetLandmarks
        });
    });

    frameNav.style.display = '';
    showFrame(currentFrameIdx);

    frameNav.querySelector('#prevFrameBtn').onclick = () => {
        if (currentFrameIdx > 0) showFrame(currentFrameIdx - 1);
    };
    frameNav.querySelector('#nextFrameBtn').onclick = () => {
        if (currentFrameIdx < poseResults.length - 1) showFrame(currentFrameIdx + 1);
    };

    statusEl.textContent = "Finished frame-based pose detection.";
}