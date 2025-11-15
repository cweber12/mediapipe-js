// pose_landmarker_frame.js
// Pose detection on extracted frames
import {
    FilesetResolver,
    PoseLandmarker,
    DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/vision_bundle.js";
import { VideoFrameExtractor } from './video_frame_extractor.js';

export async function runPoseDetectionOnFrames(videoEl, canvasEl, statusEl, poseResults, intervalSeconds, frameNav, frameCounter) {
    // Extract frames
    const extractor = new VideoFrameExtractor(videoEl, canvasEl);
    const frameDataUrls = await extractor.extractFrames(intervalSeconds);
    console.log('Extracted frames:', frameDataUrls.length);

     // Load pose model in IMAGE mode
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
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
        outputSegmentationMasks: false
    });

    poseResults.length = 0;
    let currentFrameIdx = 0;

    // Run pose detection on each frame
    for (const frameUrl of frameDataUrls) {
        const img = new Image();
        img.src = frameUrl;
        await new Promise(resolve => { img.onload = resolve; });
        canvasEl.width = img.width;
        canvasEl.height = img.height;
        const ctx = canvasEl.getContext('2d');
        ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
        ctx.drawImage(img, 0, 0, canvasEl.width, canvasEl.height);
        const result = poseLandmarker.detect(img);
        if (result.landmarks && result.landmarks.length > 0) {
            const drawingUtils = new DrawingUtils(ctx);
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
        // Save processed image
        const processedFrameUrl = canvasEl.toDataURL();
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
