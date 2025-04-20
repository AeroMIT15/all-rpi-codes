import cv2
import numpy as np
import os
import time

# Ensure the output folder exists
os.makedirs("aerothon4", exist_ok=True)
os.makedirs("red_detects", exist_ok=True)

def capture_frame():
    # Capture a frame using libcamera-still command (specific to Raspberry Pi, adjust if using another setup)
    os.system('libcamera-still -n -o frame.jpg --immediate --width 1920 --height 1080 --timeout 1')
    frame = cv2.imread('frame.jpg')
    return frame

def detect_shapes(frame):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define a narrower red color range to avoid detecting orange
    lower_red1 = np.array([0, 100, 100])  # Start of red range
    upper_red1 = np.array([10, 255, 255])  # End of red range
    lower_red2 = np.array([170, 100, 100])  # Start of red range (wrap around)
    upper_red2 = np.array([180, 255, 255])  # End of red range (wrap around)

    # Create masks for red (both ranges)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    # Apply morphological operations to clean the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(r"red_detects/shapes_detected.jpg", mask)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Minimum area threshold to ignore small contours
    min_area = 100  # Set a threshold value for the minimum area of a contour

    # Initialize counters for shapes
    triangle_count = 0
    square_count = 0
    rectangle_count = 0

    # Iterate through contours and detect shapes
    for contour in contours:
        # Ignore contours that are too small
        if cv2.contourArea(contour) < min_area:
            continue

        # Approximate the contour to a polygon with fewer vertices
        epsilon = 0.04 * cv2.arcLength(contour, True)  # Adjust epsilon to fine-tune the approximation
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Detect triangle (3 vertices)
        if len(approx) == 3:
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 5)  # Green for triangles
            triangle_count += 1
        # Detect square or rectangle (4 vertices)
        elif len(approx) == 4:
            # Check if the polygon is a square or a rectangle
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if abs(aspect_ratio - 1) <= 0.1:
                # It's a square if aspect ratio is close to 1
                cv2.drawContours(frame, [approx], 0, (0, 0, 255), 5)  # Red for square
                square_count += 1
            else:
                # It's a rectangle if aspect ratio is not 1
                cv2.drawContours(frame, [approx], 0, (255, 0, 0), 5)  # Blue for rectangle
                rectangle_count += 1

    # Display the count of detected shapes
    print(f"Detected Triangles: {triangle_count}")
    print(f"Detected Squares: {square_count}")
    print(f"Detected Rectangles: {rectangle_count}")

    # Generate a unique filename for each frame with shapes
    timestamp = int(time.time())
    frame_filename = f"aerothon4/shapes_detected_{timestamp}.jpg"
    
    # Save the image with detected shapes
    cv2.imwrite(frame_filename, frame)

def main():
    while True:
        # Capture a frame using libcamera-still
        frame = capture_frame()
        if frame is None:
            print("Error: Could not capture image.")
            break

        # Run the detection on the captured frame
        detect_shapes(frame)

        # Optional: Add a small delay to avoid overwhelming the system (e.g., 1 second)
        time.sleep(1)

if __name__ == "__main__":
    main()