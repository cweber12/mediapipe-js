// pose_landmarker_frame.js
// Pose detection on extracted frames

import {
    FilesetResolver,
    PoseLandmarker, 
    DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/vision_bundle.js";
import { VideoFrameExtractor } from './video_frame_extractor.js';

export async function runPoseDetectionOnFrames(
    videoEl, canvasEl, statusEl, poseResults, intervalSeconds, frameNav, frameCounter, cropRect
) {
    const extractor = new VideoFrameExtractor(videoEl, canvasEl);
    const frameDataUrls = await extractor.extractFrames(intervalSeconds);

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
    let currentFrameIdx = 0;
    let crop = cropRect ? { ...cropRect } : null;

    for (let i = 0; i < frameDataUrls.length; i++) {
        const frameUrl = frameDataUrls[i];
        const img = new Image();
        img.src = frameUrl;
        await new Promise(resolve => { img.onload = resolve; });

        // Crop logic
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

        // For subsequent frames, center crop on hip midpoint
        if (crop && result.landmarks && result.landmarks.length > 0) {
            const landmarkSet = result.landmarks[0];
            const leftHipIdx = 23;
            const rightHipIdx = 24;
            if (landmarkSet[leftHipIdx] && landmarkSet[rightHipIdx]) {
                const centerX = ((landmarkSet[leftHipIdx].x + landmarkSet[rightHipIdx].x) / 2) * crop.width;
                const centerY = ((landmarkSet[leftHipIdx].y + landmarkSet[rightHipIdx].y) / 2) * crop.height;
                crop.left = Math.max(0, Math.round(crop.left + centerX - crop.width / 2));
                crop.top = Math.max(0, Math.round(crop.top + centerY - crop.height / 2));
            }
        }

        // Draw landmarks on original frame canvas, offset by crop
        canvasEl.width = img.width;
        canvasEl.height = img.height;
        const ctx = canvasEl.getContext('2d');
        ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
        ctx.drawImage(img, 0, 0, img.width, img.height);

        if (result.landmarks && result.landmarks.length > 0) {
            for (const landmarkSet of result.landmarks) {
                // Offset each landmark by crop.left/crop.top
                const offsetLandmarks = landmarkSet.map(lm => ({
                    ...lm,
                    x: (lm.x * crop.width + crop.left) / img.width,
                    y: (lm.y * crop.height + crop.top) / img.height
                }));
                // Draw on original frame
                offsetLandmarks.forEach(lm => {
                    ctx.beginPath();
                    ctx.arc(lm.x * img.width, lm.y * img.height, 4, 0, 2 * Math.PI);
                    ctx.fillStyle = 'lime';
                    ctx.fill();
                });
                // Optionally, draw connectors (lines between landmarks)
                // You can use PoseLandmarker.POSE_CONNECTIONS and draw lines between offsetLandmarks
            }
            // Save landmark data with original frame coordinates
            poseResults.push({
                frameIdx: i,
                frameUrl,
                landmarks: result.landmarks.map(landmarkSet =>
                    landmarkSet.map(lm => ({
                        x: lm.x * crop.width + crop.left,
                        y: lm.y * crop.height + crop.top,
                        z: lm.z,
                        visibility: lm.visibility
                    }))
                )
            });
        } else {
            poseResults.push({
                frameIdx: i,
                frameUrl,
                landmarks: []
            });
        }
    }

    // Frame navigation
    function showFrame(idx) {
        if (!poseResults.length) return;
        currentFrameIdx = Math.max(0, Math.min(idx, poseResults.length - 1));
        const frameData = poseResults[currentFrameIdx];
        const img = new Image();
        img.src = frameData.frameUrl;
        img.onload = () => {
            canvasEl.width = img.width;
            canvasEl.height = img.height;
            const ctx = canvasEl.getContext('2d');
            ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
            ctx.drawImage(img, 0, 0, img.width, img.height);
            // Draw landmarks for this frame
            if (frameData.landmarks && frameData.landmarks.length > 0) {
                frameData.landmarks.forEach(landmarkSet => {
                    landmarkSet.forEach(lm => {
                        ctx.beginPath();
                        ctx.arc(lm.x, lm.y, 4, 0, 2 * Math.PI);
                        ctx.fillStyle = 'lime';
                        ctx.fill();
                    });
                });
            }
            frameCounter.textContent = `Frame ${currentFrameIdx + 1} / ${poseResults.length}`;
        };
    }

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