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
    # Use `preexec_fn=os.setsid` so we can kill the whole process group later
    return subprocess.Popen(cmd, shell=True, preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))

def save_detected_image(frame, count):
    # Save the frame to the images folder on the Desktop
    image_folder = '/home/adr123/Desktop/images'
    os.makedirs(image_folder, exist_ok=True)  # Create the folder if it doesn't exist
    image_path = os.path.join(image_folder, f"detected_{count}.jpg")
    cv2.imwrite(image_path, frame)
    print(f"Image saved: {image_path}")

def main():
    # Start virtual cam pipeline
    virtual_cam_process = start_virtual_cam()
    time.sleep(1.5)  # Give subprocess time to initialize at least

    # Poll for /dev/video10 to become readable
    for i in range(20):  # Try for ~10 seconds
        cap = cv2.VideoCapture("/dev/video10")
        if cap.isOpened():
            break
        print(f"Waiting for /dev/video10... ({i+1})")
        time.sleep(0.5)

    if not cap.isOpened():
        print("❌ Error: Could not open /dev/video10 after waiting.")
        virtual_cam_process.terminate()
        return

    # Load your trained YOLO model
    model = YOLO('/home/adr123/Desktop/best.pt')
    conf_threshold = 0.5

    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Press 'q' to quit.")
    detection_count = 0  # To keep track of how many images are saved
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Start time for FPS calculation
        start_time = time.time()

        # Perform detection
        results = model(frame, conf=conf_threshold)

        # Check if any targets are detected (i.e., results have detections)
        if len(results[0].boxes) > 0:  # If detections exist
            detection_count += 1
            save_detected_image(frame, detection_count)

        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)

        # Draw bounding boxes and labels on frame
        annotated_frame = results[0].plot()

        # Add FPS text to the frame
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the result
        cv2.imshow("Target Detection", annotated_frame)

        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Cleanly stop the virtual cam pipeline
    virtual_cam_process.terminate()
    print("Stopped virtual cam.")

if __name__ == "__main__":
    main()
