
# RPI COMMANDS

This guide sets up a virtual environment, streams the Raspberry Pi camera to a virtual video device, and runs a Python script that uses the virtual camera.
---

## 📦NOTE: FOR THE NEW CODE
just make virtual environment and run targetfinal.py from desktop

---

## 📦 Step 1: Create a Virtual Environment

```bash
python3 -m venv tflite-env
source tflite-env/bin/activate
```

---



## ▶️ Step 2: Run Your Python Script

```bash
cd ~/Desktop
python targetfinal.py
```

---

## ✅ Notes

- Ensure `libcamera`, `v4l2loopback`, and `ffmpeg` are installed on your system.
- Keep the streaming command running while executing the Python script that uses the virtual camera.


