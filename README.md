# all-rpi-codes




#create a virtual environment:

python3 -m venv tflite-env
source tflite-env/bin/activate


#use licamera virtual device:

sudo modprobe v4l2loopback video_nr=10 card_label="VirtualCam" exclusive_caps=1

#pipe libcamera-vid to the virtual camera:

libcamera-vid -t 0 --inline --codec mjpeg -o - | ffmpeg -i - -f v4l2 -vcodec mjpeg /dev/video10

#Run the code

cd Desktop

python /home/adr2025/Desktop/target.py
