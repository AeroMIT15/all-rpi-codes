import cv2
import time
import subprocess
import signal
from ultralytics import YOLO
import os

def start_virtual_cam():
    # Start the libcamera-vid | ffmpeg pipeline as a background process
    cmd = (
        "libcamera-vid -t 0 --inline --codec mjpeg -o - | "
        "ffmpeg -loglevel error -i - -f v4l2 -vcodec mjpeg /dev/video10"
    )
    return subprocess.Popen(cmd, shell=True, preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))

def save_detected_image(frame, count):
    # Save the frame to the images folder on the Desktop
    image_folder = '/home/adr123/Desktop/images'
    os.makedirs(image_folder, exist_ok=True)
    image_path = os.path.join(image_folder, f"detected_{count}.jpg")
    cv2.imwrite(image_path, frame)
    print(f"Image saved: {image_path}")

def main():
    # Start virtual cam pipeline
    virtual_cam_process = start_virtual_cam()
    time.sleep(1.5)

    # Wait for /dev/video10 to be ready
    for i in range(20):
        cap = cv2.VideoCapture("/dev/video10")
        if cap.isOpened():
            break
        print(f"Waiting for /dev/video10... ({i+1})")
        time.sleep(0.5)

    if not cap.isOpened():
        print("❌ Error: Could not open /dev/video10 after waiting.")
        virtual_cam_process.terminate()
        return

    # Load YOLO model
    model_path = '/home/adr123/Desktop/best (1).pt'
    model = YOLO(model_path)
    conf_threshold = 0.5

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    print("Press 'q' to quit.")
    detection_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        start_time = time.time()
        results = model(frame, conf=conf_threshold)

        # Save frame if objects are detected
        if len(results[0].boxes) > 0:
            detection_count += 1
            save_detected_image(frame, detection_count)

        fps = 1.0 / (time.time() - start_time)
        annotated_frame = results[0].plot()

        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Target Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    virtual_cam_process.terminate()
    print("Stopped virtual cam.")

if __name__ == "__main__":
    main()
