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
    // Initialize frame extractor
    const extractor = new VideoFrameExtractor(videoEl, canvasEl);
    // Extract frames at specified interval
    const frameDataUrls = await extractor.extractFrames(intervalSeconds);
    console.log('Extracted frames:', frameDataUrls.length);

    // Load pose model in IMAGE mode
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );
    // Create PoseLandmarker instance
    const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            delegate: "GPU"
        },
        runningMode: "IMAGE",
        numPoses: 1,
        minPoseDetectionConfidence: 0.9,
        minPosePresenceConfidence: 0.9,
        minTrackingConfidence: 0.9,
        outputSegmentationMasks: false
    });

    // Clear previous results
    poseResults.length = 0;
    let currentFrameIdx = 0;

    // Track crop for subsequent frames
    let crop = cropRect ? { ...cropRect } : null;

    // Run pose detection on each frame
    for (const frameUrl of frameDataUrls) {
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
            // Find left and right hip indices
            const leftHipIdx = PoseLandmarker.LANDMARK_NAMES.indexOf('left_hip');
            const rightHipIdx = PoseLandmarker.LANDMARK_NAMES.indexOf('right_hip');
            if (landmarkSet[leftHipIdx] && landmarkSet[rightHipIdx]) {
                const centerX = ((landmarkSet[leftHipIdx].x + landmarkSet[rightHipIdx].x) / 2) * croppedImg.width;
                const centerY = ((landmarkSet[leftHipIdx].y + landmarkSet[rightHipIdx].y) / 2) * croppedImg.height;
                crop.left = Math.max(0, Math.round(centerX - crop.width / 2));
                crop.top = Math.max(0, Math.round(centerY - crop.height / 2));
            }
        }

        // Draw landmarks on cropped canvas
        if (result.landmarks && result.landmarks.length > 0) {
            const drawingUtils = new DrawingUtils(croppedCtx);
            for (const landmarkSet of result.landmarks) {
                drawingUtils.drawLandmarks(landmarkSet, {
                    radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1)
                });
                drawingUtils.drawConnectors(
                    landmarkSet,
                    PoseLandmarker.POSE_CONNECTIONS,
                    { color: 'lime', lineWidth: 4 }
                );
            }
        }
        const processedFrameUrl = croppedCanvas.toDataURL();
        poseResults.push({ frameUrl: processedFrameUrl, landmarks: result.landmarks });
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
            ctx.drawImage(img, 0, 0, canvasEl.width, canvasEl.height);
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