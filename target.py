import cv2
import time
from ultralytics import YOLO

def main():
    # Load your trained model
    model_path = '/home/adr123/Desktop/best.pt'  # Path to your trained model
    model = YOLO(model_path)
    
    # Set confidence threshold
    conf_threshold = 0.5
    
    # Open webcam
    cap = cv2.VideoCapture("/dev/video10")

    
    # Check if webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set webcam resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Press 'q' to quit.")
    
    # Loop through frames
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        
        # Start time for FPS calculation
        start_time = time.time()
        
        # Perform detection
        results = model(frame, conf=conf_threshold)
        
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

if __name__ == "__main__":
    main()