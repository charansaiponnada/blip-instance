#!/usr/bin/env python3
"""Simple webcam captioner with read-aloud.

This version only uses the local webcam (cv2.VideoCapture with index) and speaks
captions with pyttsx3. It keeps model loading lazy.

Usage:
  python capture_and_caption.py --model-dir final_model --device cpu --camera-index 0

Press 'q' to quit.
"""

import argparse
import sys
import time
from pathlib import Path

try:
    import cv2
except Exception:
    print("Please install opencv-python to use this script.")
    raise

try:
    from PIL import Image
except Exception:
    print("Please install Pillow to use this script.")
    raise

import threading


class TTSSpeaker:
    def __init__(self):
        try:
            import pyttsx3
        except Exception:
            self.engine = None
            return
        self.engine = pyttsx3.init()
        self.lock = threading.Lock()

    def speak(self, text: str):
        if not self.engine or not text:
            return
        # run in background thread to avoid blocking
        def _s():
            with self.lock:
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception:
                    pass

        t = threading.Thread(target=_s, daemon=True)
        t.start()


def load_model(model_dir: str, device: str):
    from transformers import BlipForConditionalGeneration, BlipProcessor
    import torch

    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    processor = BlipProcessor.from_pretrained(model_dir, local_files_only=True)
    model = BlipForConditionalGeneration.from_pretrained(model_dir, local_files_only=True)
    model.to(device)
    model.eval()
    return processor, model


def generate_caption(processor, model, pil_img, device, max_length=50):
    import torch
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_length)
    caption = processor.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
    return caption


def draw_caption(frame, caption: str):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    # simple single-line caption at bottom
    cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, caption, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Simple webcam captioner with TTS")
    parser.add_argument("--model-dir", default="final_model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=50)
    args = parser.parse_args()

    device = args.device

    try:
        processor, model = load_model(args.model_dir, device)
    except Exception as e:
        print("Failed to load model:", e)
        sys.exit(1)

    tts = TTSSpeaker()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("Could not open local camera index", args.camera_index)
        sys.exit(1)

    last_caption = ""
    last_time = 0.0
    cooldown = 1.5

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break

            now = time.time()
            if now - last_time > cooldown:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                try:
                    caption = generate_caption(processor, model, pil, device, max_length=args.max_length)
                except Exception as e:
                    caption = f"(error: {e})"
                last_caption = caption
                last_time = now
                print("Caption:", caption)
                tts.speak(caption)

            out = draw_caption(frame, last_caption)
            cv2.imshow("Captioner - q to quit", out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Capture webcam frames and generate captions using a local BLIP model.

Place your finetuned BLIP model files in a directory (default: ./final_model).
Usage examples:
  python capture_and_caption.py --model-dir final_model --device cpu
  python capture_and_caption.py --model-dir final_model --device cuda

The script shows the webcam feed with the generated caption overlaid.
Press 'q' to quit.
"""

import argparse
import time
import sys
from pathlib import Path

try:
    import cv2
except Exception as e:
    print("Error importing OpenCV (cv2). Make sure you installed opencv-python.")
    raise

try:
    from PIL import Image
except Exception:
    print("Error importing Pillow. Install pillow via pip.")
    raise

import torch




def load_model(model_dir: str, device: str):
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Loading processor and model from: {model_dir}")
    # Import transformers classes lazily to avoid import-time side-effects when e.g. using --help
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(model_dir, local_files_only=True)
    model = BlipForConditionalGeneration.from_pretrained(model_dir, local_files_only=True)

    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            model = model.to(device)
            # try to use half precision on CUDA to save memory
            model.half()
            print("Moved model to CUDA and converted to float16 (half) to save memory.")
        except Exception:
            print("Warning: Couldn't convert model to half or move to CUDA; using default dtype.")
            model = model.to(device)
    else:
        model = model.to(device)

    model.eval()
    return processor, model


def generate_caption(processor, model, image_pil, device: torch.device, max_length=50):
    # Prepare inputs
    inputs = processor(images=image_pil, return_tensors="pt")
    # Move tensors to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate (adjust parameters if you want different behaviour)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_length, num_beams=4, do_sample=False)

    # Decode
    caption = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return caption


def draw_caption(frame, caption: str):
    # Draw a semi-opaque rectangle and put caption text on it.
    overlay = frame.copy()
    h, w = frame.shape[:2]
    # Wrap text if too long
    max_width = 60
    words = caption.split()
    lines = []
    cur = []
    for wword in words:
        cur.append(wword)
        if len(" ".join(cur)) > max_width:
            cur.pop()
            lines.append(" ".join(cur))
            cur = [wword]
    if cur:
        lines.append(" ".join(cur))

    # compute rectangle size
    padding = 10
    line_height = 20
    rect_h = padding * 2 + line_height * len(lines)
    rect_w = w - 20

    # position rectangle at bottom
    rect_x1 = 10
    rect_y1 = h - rect_h - 10
    rect_x2 = rect_x1 + rect_w
    rect_y2 = rect_y1 + rect_h

    # overlay rectangle
    alpha = 0.6
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # draw text lines
    y = rect_y1 + padding + 15
    for line in lines:
        cv2.putText(frame, line, (rect_x1 + padding, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += line_height

    return frame


def main():
    parser = argparse.ArgumentParser(description="Webcam captioning with a local BLIP model")
    parser.add_argument("--model-dir", type=str, default="final_model", help="Path to folder containing the finetuned BLIP model files")
    parser.add_argument("--device", type=str, default=None, help="Device to run the model on. E.g. cpu or cuda. If omitted, cuda will be used if available.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index for cv2.VideoCapture")
    parser.add_argument("--camera-url", type=str, default=None, help="URL for an IP camera / mobile webcam stream (e.g. http://192.168.1.5:8080/video). If set, this is used instead of --camera-index")
    parser.add_argument("--dshow-name", type=str, default=None, help="(Windows) DirectShow device name, e.g. 'DroidCam Source 3' or 'Your Phone' - use with cv2 CAP_DSHOW")
    parser.add_argument("--auto-scan", action="store_true", help="If set, automatically scan local camera indices to find a working camera when the specified one fails.")
    parser.add_argument("--max-length", type=int, default=50, help="Max generated tokens for caption")
    args = parser.parse_args()

    # Decide device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    try:
        processor, model = load_model(args.model_dir, str(device))
    except Exception as e:
        print("Failed to load model:", e)
        sys.exit(1)

    # Open camera: prefer URL if provided (for mobile webcam apps that expose an MJPEG/RTSP stream)
    if args.camera_url:
        print(f"Opening camera stream from URL: {args.camera_url}")
        cap = cv2.VideoCapture(args.camera_url)
        # retry a few times if the stream isn't immediately available
        open_tries = 0
        while not cap.isOpened() and open_tries < 5:
            open_tries += 1
            print(f"Stream not ready, retrying ({open_tries}/5) ...")
            time.sleep(1)
            cap.open(args.camera_url)
        if not cap.isOpened():
            print("Error: Could not open camera URL. Check that the mobile webcam app is running and the URL is correct.")
            sys.exit(1)
    else:
        # If a DirectShow device name is given, try opening it directly (Windows)
        def try_open_index(idx, use_dshow=False):
            try:
                if use_dshow:
                    c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                else:
                    c = cv2.VideoCapture(idx)
                if c.isOpened():
                    return c
                c.release()
            except Exception:
                pass
            return None

        cap = None
        if args.dshow_name:
            # Try opening by DirectShow device name (prefix with 'video=')
            try:
                print(f"Trying DirectShow device name: {args.dshow_name}")
                cap = cv2.VideoCapture(f"video={args.dshow_name}", cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    cap = None
            except Exception:
                cap = None

        if cap is None:
            cap = try_open_index(args.camera_index)

        if (cap is None) or (not cap.isOpened()):
            if args.auto_scan:
                print("Initial camera open failed; auto-scanning local camera indices 0..8 to find a working device...")
                found = None
                for i in range(0, 9):
                    print(f"Trying index {i} (CAP_DSHOW)...")
                    c = try_open_index(f"video={i}", use_dshow=True)
                    if c is None:
                        # try numeric index without dshow
                        c = try_open_index(i, use_dshow=False)
                    if c is not None and c.isOpened():
                        print(f"Found camera at index {i}")
                        cap = c
                        found = i
                        break
                if cap is None or not cap.isOpened():
                    print("Auto-scan failed: no working camera indices found. Check that your phone's webcam app or virtual webcam driver is running.")
                    sys.exit(1)
                else:
                    print(f"Using camera index {found}")
            else:
                print("Error: Could not open local webcam/camera. Make sure camera index is correct and no other app is using it. Use --auto-scan to try scanning indices.")
                sys.exit(1)

    last_caption = ""
    caption_timestamp = 0.0
    caption_cooldown = 1.0  # seconds between caption generation

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break

            now = time.time()
            # Generate caption at most once per cooldown
            if now - caption_timestamp > caption_cooldown:
                # Convert BGR to RGB and to PIL
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                try:
                    caption = generate_caption(processor, model, pil_img, device, max_length=args.max_length)
                    last_caption = caption
                except Exception as e:
                    last_caption = f"(error generating caption: {e})"
                caption_timestamp = now

            out = draw_caption(frame, last_caption)
            cv2.imshow("Webcam - press q to quit", out)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
