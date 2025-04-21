
# RPI COMMANDS

This guide sets up a virtual environment, streams the Raspberry Pi camera to a virtual video device, and runs a Python script that uses the virtual camera.

---

## 📦 Step 1: Create a Virtual Environment

```bash
python3 -m venv tflite-env
source tflite-env/bin/activate
```

---

## 🎥 Step 2: Enable a Virtual Camera Device

```bash
sudo modprobe v4l2loopback video_nr=10 card_label="VirtualCam" exclusive_caps=1
```

---

## 🔄 Step 3: Stream from Pi Camera to Virtual Device

```bash
libcamera-vid -t 0 --inline --codec mjpeg -o - | ffmpeg -i - -f v4l2 -vcodec mjpeg /dev/video10
```

This will stream the Pi camera output to `/dev/video10`.

---

## ▶️ Step 4: Run Your Python Script

```bash
cd ~/Desktop
python /home/adr2025/Desktop/target.py
```

---

## ✅ Notes

- Ensure `libcamera`, `v4l2loopback`, and `ffmpeg` are installed on your system.
- Keep the streaming command running while executing the Python script that uses the virtual camera.


