// pose_landmarker.js
// Main JS code for MediaPipe Pose Landmarker on video to JSON export

import {
      FilesetResolver, // for loading the model
      PoseLandmarker, // the pose landmarker class
      DrawingUtils // utility for drawing landmarks and connectors
    } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/vision_bundle.js";

    const videoFileInput = document.getElementById('videoFile'); // file input element
    const videoEl        = document.getElementById('video'); // video element
    const canvasEl       = document.getElementById('overlay'); // canvas for overlay
    const startBtn       = document.getElementById('startBtn'); // start button
    const stopBtn        = document.getElementById('stopBtn'); // stop button
    const downloadBtn    = document.getElementById('downloadBtn'); // download button
    const statusEl       = document.getElementById('status'); // status display

    let poseLandmarker = null; // will hold the pose landmarker instance
    let drawingUtils   = null; // utility for drawing landmarks and connectors
    let canvasCtx      = null; // canvas 2D context

    let running = false; // flag to control processing loop
    let lastVideoTime = -1; // last processed video time

    // Collected results: one entry per processed frame
    const poseResults = [];

    // Load the pose model
    async function initPoseLandmarker() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          // WASM path
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            // You can switch to full / heavy if you want higher accuracy
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
          outputSegmentationMasks: false
        });

        canvasCtx    = canvasEl.getContext('2d');
        drawingUtils = new DrawingUtils(canvasCtx);

        statusEl.textContent = "Model loaded. Choose a video file to begin.";
      } catch (err) {
        console.error(err);
        statusEl.textContent = "Error loading pose model. Check console.";
      }
    }

    initPoseLandmarker();

    // When a video file is selected
    videoFileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const url = URL.createObjectURL(file);
      videoEl.src = url;

      // Reset previous results
      poseResults.length = 0;
      downloadBtn.disabled = true;

      statusEl.textContent = "Video loaded. Press 'Start Pose Detection'.";
      startBtn.disabled = false;
    });

    // Sync canvas to video size once metadata is available
    videoEl.addEventListener('loadedmetadata', () => {
      canvasEl.width  = videoEl.videoWidth;
      canvasEl.height = videoEl.videoHeight;
    });

    startBtn.addEventListener('click', () => {
      if (!poseLandmarker) {
        alert("Pose model not ready yet.");
        return;
      }
      if (!videoEl.src) {
        alert("Please select a video first.");
        return;
      }

      running = true;
      lastVideoTime = -1;
      poseResults.length = 0; // clear old data

      startBtn.disabled = true;
      stopBtn.disabled = false;
      downloadBtn.disabled = true;

      statusEl.textContent = "Running pose detection… (this may be slow for long videos)";
      // Ensure playback starts
      videoEl.play();
      requestAnimationFrame(processVideoFrame);
    });

    stopBtn.addEventListener('click', () => {
      running = false;
      startBtn.disabled = false;
      stopBtn.disabled = true;
      downloadBtn.disabled = poseResults.length === 0;

      statusEl.textContent = "Stopped. You can download JSON or start again.";
    });

    function processVideoFrame() {
      if (!running || !poseLandmarker) return;

      // If video ended, stop
      if (videoEl.ended || videoEl.currentTime >= videoEl.duration) {
        running = false;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        downloadBtn.disabled = poseResults.length === 0;
        statusEl.textContent = "Finished processing video.";
        return;
      }

      const now = performance.now();

      // Only process when the video's currentTime advanced
      if (videoEl.currentTime !== lastVideoTime) {
        lastVideoTime = videoEl.currentTime;

        poseLandmarker.detectForVideo(videoEl, now, (result) => {
          // Clear overlay
          canvasCtx.clearRect(0, 0, canvasEl.width, canvasEl.height);

          if (result && result.landmarks && result.landmarks.length > 0) {
            const frameWidth  = videoEl.videoWidth;
            const frameHeight = videoEl.videoHeight;

            // Draw landmarks on canvas overlay
            for (const landmarkSet of result.landmarks) {
              drawingUtils.drawLandmarks(landmarkSet, {
                radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1)
              });
              drawingUtils.drawConnectors(
                landmarkSet,
                PoseLandmarker.POSE_CONNECTIONS
              );

              // Convert normalized coords (0–1) to pixel coords and store
              const pixels = landmarkSet.map((lm, index) => ({
                index,
                x_norm: lm.x,
                y_norm: lm.y,
                x_px: lm.x * frameWidth,
                y_px: lm.y * frameHeight,
                z: lm.z,
                visibility: lm.visibility
              }));

              poseResults.push({
                timeSec: videoEl.currentTime,
                frameWidth,
                frameHeight,
                landmarks: pixels
              });
            }
          }
        });
      }

      // Continue loop
      requestAnimationFrame(processVideoFrame);
    }

    downloadBtn.addEventListener('click', () => {
      if (poseResults.length === 0) {
        alert("No pose data collected yet.");
        return;
      }

      const blob = new Blob(
        [JSON.stringify(poseResults, null, 2)],
        { type: "application/json" }
      );

      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = "pose_landmarks.json";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    });

    // Clean up on page unload
    window.addEventListener('beforeunload', () => {
      if (poseLandmarker) {
        poseLandmarker.close();
      }
    });