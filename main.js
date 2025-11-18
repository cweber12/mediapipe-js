// main.js
// Main entry point: choose mode and run pose detection
import { runPoseDetectionOnVideo } from './pose_landmarker_video.js';
import { runPoseDetectionOnFrames } from './pose_landmarker_frame.js';
import { VideoFrameExtractor } from './video_frame_extractor.js';
import { setupCropBox } from './setup_crop_box.js';

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
const cropBoxEl      = document.getElementById('cropBox');

// Results
const poseResults = [];

// Mode selection logic

videoEl.addEventListener('loadedmetadata', () => {
  videoDetectBtn.disabled = false;
  frameDetectBtn.disabled = false;

  // Adjust canvas pixel buffer size
  canvasEl.width = videoEl.videoWidth;
  canvasEl.height = videoEl.videoHeight;
  canvasEl.style.position = 'absolute';
  canvasEl.style.left = '0px';
  canvasEl.style.top = '0px';
  canvasEl.style.pointerEvents = 'none'; // allow clicks to pass through to video

  // Set canvas and crop box size to match video display size
  const videoRect = videoEl.getBoundingClientRect();
  canvasEl.style.width = videoRect.width + 'px';
  canvasEl.style.height = videoRect.height + 'px';
  cropBoxEl.style.width = videoRect.width + 'px';
  cropBoxEl.style.height = videoRect.height + 'px';
  cropBoxEl.style.left = '0px';
  cropBoxEl.style.top = '0px';

  statusEl.textContent = "Video loaded. Choose detection mode.";
});

window.addEventListener('resize', () => {
  const videoRect = videoEl.getBoundingClientRect();
  canvasEl.style.width = videoRect.width + 'px';
  canvasEl.style.height = videoRect.height + 'px';
  cropBoxEl.style.width = videoRect.width + 'px';
  cropBoxEl.style.height = videoRect.height + 'px';
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

frameDetectBtn.addEventListener('click', async function handleFrameDetect() {
  intervalLabel.style.display = '';
  frameNav.style.display = 'none';
  canvasEl.style.display = '';
  videoEl.style.display = '';
  videoEl.style.position = 'relative';

  // Pause video and seek to first frame
  videoEl.pause();
  videoEl.currentTime = 0;
  await new Promise(resolve => {
    videoEl.onseeked = resolve;
  });

  // Show crop box over video
  cropBoxEl.hidden = false;
  cropBoxEl.style.left = '0px';
  cropBoxEl.style.top = '0px';
  cropBoxEl.style.width = videoEl.videoWidth + 'px';
  cropBoxEl.style.height = videoEl.videoHeight + 'px';
  setupCropBox(videoEl, cropBoxEl);

  statusEl.textContent = "Adjust crop box, then click Detect from Frames again to confirm crop and start detection.";

  // Replace this handler with a one-time confirm handler
  frameDetectBtn.removeEventListener('click', handleFrameDetect);
  frameDetectBtn.addEventListener('click', async function confirmCropHandler() {
    cropBoxEl.hidden = true;
    const cropRect = {
      left: parseInt(cropBoxEl.style.left, 10),
      top: parseInt(cropBoxEl.style.top, 10),
      width: parseInt(cropBoxEl.style.width, 10),
      height: parseInt(cropBoxEl.style.height, 10)
    };
    const n = parseInt(intervalInput.value, 10) || 1;
    poseResults.length = 0;
    downloadBtn.disabled = true;
    await runPoseDetectionOnFrames(
      videoEl,
      canvasEl,
      statusEl,
      poseResults,
      n,
      frameNav,
      frameCounter,
      cropRect
    );
    downloadBtn.disabled = poseResults.length === 0;
    frameDetectBtn.removeEventListener('click', confirmCropHandler);
    frameDetectBtn.addEventListener('click', handleFrameDetect);
  }, { once: true });
});

videoDetectBtn.addEventListener('click', () => {
  intervalLabel.style.display = 'none';
  frameNav.style.display = 'none';
  canvasEl.style.display = '';
  canvasEl.style.position = 'absolute'; 
  videoEl.style.display = '';
  runVideoDetection();
});

async function runVideoDetection() {
    await runPoseDetectionOnVideo(
        videoEl, 
        canvasEl, 
        statusEl, 
        poseResults, 
        videoDetectBtn, 
        stopBtn, 
        downloadBtn
    );

}

async function runFrameDetection() {
    const n = parseInt(intervalInput.value, 10) || 1;
    poseResults.length = 0; 
    downloadBtn.disabled = true; 
    await runPoseDetectionOnFrames(
        videoEl, 
        canvasEl, 
        statusEl, 
        poseResults, 
        n, 
        frameNav, 
        frameCounter
    );
    downloadBtn.disabled = poseResults.length === 0;
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
