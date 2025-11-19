#!/usr/bin/env python3
"""
Multi-threaded BLIP webcam captioner with independent TTS thread.

- Caption generation runs in a background thread.
- TTS runs in a separate background thread with a queue.
- Webcam feed is non-blocking.
- Ensures audio never stops even if caption generation is slow.
"""

import argparse
import sys
import time
from pathlib import Path
from queue import Queue, Empty
import threading

try:
    import cv2
except ImportError:
    print("Please install opencv-python")
    raise

try:
    from PIL import Image
except ImportError:
    print("Please install Pillow")
    raise

import torch


# ----------------- TTS QUEUE -----------------
class TTSAudioThread(threading.Thread):
    """Thread for non-blocking TTS using pyttsx3"""
    def __init__(self, rate=160):
        super().__init__(daemon=True)
        import pyttsx3
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.queue = Queue()
        self.running = True
        self.lock = threading.Lock()

    def speak(self, text: str):
        """Add text to TTS queue"""
        if text:
            self.queue.put(text)

    def run(self):
        while self.running:
            try:
                text = self.queue.get(timeout=0.1)
            except Empty:
                continue
            with self.lock:
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception:
                    pass


# ----------------- BLIP -----------------
def load_model(model_dir: str, device: str):
    from transformers import BlipForConditionalGeneration, BlipProcessor
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    processor = BlipProcessor.from_pretrained(model_dir, local_files_only=True)
    model = BlipForConditionalGeneration.from_pretrained(model_dir, local_files_only=True)
    model.to(device)
    model.eval()
    return processor, model


def generate_caption(processor, model, pil_img, device, max_length=50):
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_length)
    caption = processor.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
    return caption


def draw_caption(frame, caption: str):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, caption, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


# ----------------- CAPTION WORKER -----------------
class CaptionWorker(threading.Thread):
    """Background caption generator thread"""
    def __init__(self, processor, model, device, tts_thread=None, max_length=50, min_speak_interval=1.5):
        super().__init__(daemon=True)
        self.processor = processor
        self.model = model
        self.device = device
        self.max_length = max_length
        self.queue = Queue(maxsize=1)
        self.current_caption = ""
        self.lock = threading.Lock()
        self.running = True
        self.tts_thread = tts_thread
        self.min_speak_interval = min_speak_interval
        self.last_speak_time = 0.0

    def submit_frame(self, pil_img):
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except Empty:
                pass
        self.queue.put(pil_img)

    def get_caption(self):
        with self.lock:
            return self.current_caption

    def run(self):
        while self.running:
            try:
                pil_img = self.queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                caption = generate_caption(self.processor, self.model, pil_img, self.device, self.max_length)
            except Exception as e:
                caption = f"(error: {e})"

            with self.lock:
                self.current_caption = caption

            # Send caption to TTS queue independently
            now = time.time()
            if self.tts_thread and (now - self.last_speak_time) > self.min_speak_interval:
                self.tts_thread.speak(caption)
                self.last_speak_time = now


# ----------------- MAIN -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="final_model")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--tts-rate", type=int, default=160)
    parser.add_argument("--min-speak-interval", type=float, default=1.5)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    processor, model = load_model(args.model_dir, str(device))

    # Start TTS thread
    tts_thread = TTSAudioThread(rate=args.tts_rate)
    tts_thread.start()

    # Start caption worker
    caption_worker = CaptionWorker(
        processor, model, device, tts_thread=tts_thread,
        max_length=args.max_length, min_speak_interval=args.min_speak_interval
    )
    caption_worker.start()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"Could not open camera index {args.camera_index}")
        sys.exit(1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            caption_worker.submit_frame(pil_img)

            caption = caption_worker.get_caption()
            out = draw_caption(frame, caption)
            cv2.imshow("Webcam Captioner - q to quit", out)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        caption_worker.running = False
        tts_thread.running = False
        caption_worker.join(timeout=1.0)
        tts_thread.join(timeout=1.0)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
