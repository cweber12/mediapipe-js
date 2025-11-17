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
            // Center crop on hip midpoint if hips are detected
            if (landmarkSet[leftHipIdx] && landmarkSet[rightHipIdx]) {
                // Get hip center in cropped frame (normalized coordinates)
                const centerX_cropped = (landmarkSet[leftHipIdx].x + landmarkSet[rightHipIdx].x) / 2;
                const centerY_cropped = (landmarkSet[leftHipIdx].y + landmarkSet[rightHipIdx].y) / 2;
                // Convert to original frame coordinates
                const centerX_original = crop.left + centerX_cropped * crop.width;
                const centerY_original = crop.top + centerY_cropped * crop.height;
                // Update crop box for next frame, centered on hip center in original frame
                crop.left = Math.max(0, Math.round(centerX_original - crop.width / 2));
                crop.top = Math.max(0, Math.round(centerY_original - crop.height / 2));
                // Clamp crop to stay within original frame bounds
                crop.left = Math.min(crop.left, img.width - crop.width);
                crop.top = Math.min(crop.top, img.height - crop.height);
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
                // Draw connectors
                const drawingUtils = new DrawingUtils(ctx);
                drawingUtils.drawConnectors(
                    offsetLandmarks,
                    PoseLandmarker.POSE_CONNECTIONS,
                    { color: 'lime', lineWidth: 2 }
                );

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
                    // Draw circles (pixel coordinates)
                    landmarkSet.forEach(lm => {
                        ctx.beginPath();
                        ctx.arc(lm.x, lm.y, 4, 0, 2 * Math.PI);
                        ctx.fillStyle = 'lime';
                        ctx.fill();
                    });
                    // Prepare normalized landmarks for connectors
                    const normalizedLandmarks = landmarkSet.map(lm => ({
                        x: lm.x / canvasEl.width,
                        y: lm.y / canvasEl.height,
                        z: lm.z,
                        visibility: lm.visibility
                    }));
                    // Draw connectors using normalized coordinates
                    const drawingUtils = new DrawingUtils(ctx);
                    drawingUtils.drawConnectors(
                        normalizedLandmarks,
                        PoseLandmarker.POSE_CONNECTIONS,
                        { color: 'lime', lineWidth: 2 }
                    );
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