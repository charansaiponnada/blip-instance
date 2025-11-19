# Webcam captioning with local BLIP model

This small utility captures frames from your webcam and generates captions using a locally finetuned BLIP model (a VLM).
**Note:** To use the mobile cam, [click here](## Mobile Cam) for instructions.
Prereqs
- Python 3.8+ installed
- GPU + CUDA (optional) for faster inference. CPU will work but may be slow.
- Your finetuned model files placed in the `final_model/` directory (or specify another dir with --model-dir).

Quick start (Windows PowerShell)

1. Create and activate a venv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install torch (choose the right command for your CUDA) and other requirements

# CPU-only (fastest to type but slower runtime)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# OR GPU (example, change to match your CUDA version):
# pip install torch --index-url https://download.pytorch.org/whl/cu118

# Then install the rest
pip install -r requirements.txt

Note: If pip cannot find a compatible torch wheel for your platform, visit https://pytorch.org/get-started/locally/ and use the recommended install command.

3. Run the webcam caption script

```powershell
python capture_and_caption.py --model-dir final_model --device cpu
```

Replace `--device cpu` with `--device cuda` if you installed CUDA-enabled torch and want to run on GPU.

Controls
- Press `q` in the window to quit.

Troubleshooting
- If OpenCV fails to open the camera, make sure no other application is using it and try different camera indices (0,1,...):
  python capture_and_caption.py --camera-index 1
- If the model fails to load, confirm the directory contains model files (pytorch_model.bin or model.safetensors, config.json, tokenizer files, etc.).

Security & performance notes
- Generating captions on CPU may be slow. For smoother real-time behavior, use a CUDA-enabled GPU.
- This script loads the model once and reuses it for frames. If you want to caption only on-demand (e.g., key press), modify the loop accordingly.
 ----
## Mobile Cam
 To use the mobile cam connect using the droid cam withing the same wifi first run the [test.py](./test.py) file to check in which the cam is operating note the index change inside the code of mobile_cam.py
 This will make the use of the mobile cam via droid cam.