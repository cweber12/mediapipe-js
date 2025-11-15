// video_frame_extractor.js
// Extract frames from a video element into a canvas
export class VideoFrameExtractor {

    constructor(videoEl, canvasEl) {
        this.videoEl = videoEl; // HTMLVideoElement
        this.canvasEl = canvasEl; // HTMLCanvasElement
    }

    // Extract frames every n seconds
    async extractFrames(n) {
        if (!this.videoEl.src) throw new Error('No video loaded');
        const duration = this.videoEl.duration;
        const frames = [];
        for (let t = 0; t < duration; t += n) {
            await new Promise((resolve, reject) => {
                this.videoEl.currentTime = t;
                this.videoEl.onseeked = () => {
                    this.canvasEl.width = this.videoEl.videoWidth;
                    this.canvasEl.height = this.videoEl.videoHeight;
                    const ctx = this.canvasEl.getContext('2d');
                    ctx.drawImage(this.videoEl, 0, 0, this.canvasEl.width, this.canvasEl.height);
                    // You can clone the canvas or get the image data here
                    frames.push(this.canvasEl.toDataURL());
                    resolve();
                };
                this.videoEl.onerror = reject;
            });
        }
        return frames; // Array of data URLs for each frame
    }
}