// main.js
// Main entry point: choose mode and run pose detection
import { runPoseDetectionOnVideo } from './pose_landmarker_video.js';
import { runPoseDetectionOnFrames } from './pose_landmarker_frame.js';
import { VideoFrameExtractor } from './video_frame_extractor.js';

// DOM elements
const videoFileInput = document.getElementById('videoFile');
const videoEl        = document.getElementById('video');
const canvasEl       = document.getElementById('overlay');
const videoDetectBtn = document.getElementById('videoDetectBtn');
const frameDetectBtn = document.getElementById('frameDetectBtn');
const stopBtn        = document.getElementById('stopBtn');
const downloadBtn    = document.getElementById('downloadBtn');
const statusEl       = document.getElementById('status');
const intervalInput  = document.getElementById('intervalInput');
const intervalLabel  = document.getElementById('intervalLabel');
const frameNav       = document.getElementById('frameNav');
const frameCounter   = document.getElementById('frameCounter');

// Results
const poseResults = [];

// Mode selection logic

videoEl.addEventListener('loadedmetadata', () => {
  videoDetectBtn.disabled = false;
  frameDetectBtn.disabled = false;
  canvasEl.width = videoEl.videoWidth;
  canvasEl.height = videoEl.videoHeight;
  statusEl.textContent = "Video loaded. Choose detection mode.";
});

videoFileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  // Set video source to selected file
  const url = URL.createObjectURL(file);
  videoEl.src = url;
  videoDetectBtn.disabled = true;
  frameDetectBtn.disabled = true;
  statusEl.textContent = "Loading video...";
});

frameDetectBtn.addEventListener('click', () => {
  intervalLabel.style.display = '';
  frameNav.style.display = 'none';
  canvasEl.style.display = '';
  videoEl.style.display = 'none';
  runFrameDetection();
});

videoDetectBtn.addEventListener('click', () => {
  intervalLabel.style.display = 'none';
  frameNav.style.display = 'none';
  canvasEl.style.display = '';
  videoEl.style.display = '';
  runVideoDetection();
});

async function runVideoDetection() {
  await runPoseDetectionOnVideo(videoEl, canvasEl, statusEl, poseResults, videoDetectBtn, stopBtn, downloadBtn);
}

async function runFrameDetection() {
  const n = parseInt(intervalInput.value, 10) || 1;
  await runPoseDetectionOnFrames(videoEl, canvasEl, statusEl, poseResults, n, frameNav, frameCounter);
}


downloadBtn.addEventListener('click', () => {
  if (poseResults.length === 0) {
    alert("No pose data collected yet.");
    return;
  }
  const blob = new Blob([
    JSON.stringify(poseResults, null, 2)
  ], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = "pose_landmarks.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
});
