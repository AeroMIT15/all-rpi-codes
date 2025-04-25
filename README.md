
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

#use licamera virtual device and make it available for use:

sudo modprobe v4l2loopback video_nr=10 card_label="VirtualCam" exclusive_caps=1
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

---

## How to setup RPI with VNC

```bash
sudo raspi-config
```
then manually enable ssh and vnc to yes

prathiksha@prathiksha-Aspire-A715-79G:~$ hostname -I
10.77.181.205 172.17.0.1 2409:40f2:3017:a44c:114b:4240:2e1a:758 2409:40f2:3017:a44c:e087:b99a:930d:9bec 


then on laptop:

prathiksha@prathiksha-Aspire-A715-79G:~$ ssh  adr123@10.77.181.120
adr123@10.77.181.120's password: 
