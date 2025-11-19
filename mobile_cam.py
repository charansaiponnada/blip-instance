import cv2
import torch
import threading
from transformers import BlipProcessor, BlipForConditionalGeneration
import pyttsx3

# ====== LOAD BLIP ======
model_path = "./final_model"
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ====== SETUP NON-BLOCKING TTS ======
engine = pyttsx3.init()
engine.setProperty("rate", 160)

tts_lock = threading.Lock()

def speak_non_blocking(text):
    def run():
        with tts_lock:
            engine.say(text)
            engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()

# ====== CAMERA ======
cap = cv2.VideoCapture(1)  # your DroidCam index

if not cap.isOpened():
    print("Error: Could not access DroidCam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # BLIP preprocessing
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=img_rgb, return_tensors="pt").to(device)

    # Generate caption
    with torch.no_grad():
        out = model.generate(**inputs, max_length=30)

    caption = processor.decode(out[0], skip_special_tokens=True)

    # Speak caption (non-blocking — will work EVERY TIME)
    speak_non_blocking(caption)

    # Show caption on screen
    cv2.putText(frame, caption, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("BLIP Captioning - DroidCam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
