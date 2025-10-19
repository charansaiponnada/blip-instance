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

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("Error: Could not open webcam/camera. Make sure camera index is correct and no other app is using it.")
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
