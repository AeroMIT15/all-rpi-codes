import cv2
import numpy as np
import math

class CentroidTracker:
    def __init__(self, max_disappeared=5):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            cX = int((x + x + w) / 2.0)
            cY = int((y + y + h) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects

def calculate_frame_area(hfov_deg=82, vfov_deg=46, distance_m=0.2):
    hfov_rad = math.radians(hfov_deg)
    vfov_rad = math.radians(vfov_deg)
    width = 2 * distance_m * math.tan(hfov_rad / 2)
    height = 2 * distance_m * math.tan(vfov_rad / 2)
    area = width * height
    return width, height, area

def pixel_to_real_area(pixel_area, frame_area_m2, resolution=(1920, 1080)):
    total_pixels = resolution[0] * resolution[1]
    area_m2 = (pixel_area / total_pixels) * frame_area_m2
    return area_m2 * 10000 
    
cap = cv2.VideoCapture("/dev/video10")
cap.set(3, 1920)
cap.set(4, 1080)

color_ranges = {
    "Red1":   (np.array([0, 140, 70], dtype=np.uint8),  np.array([10, 255, 255], dtype=np.uint8)),
    "Red2":   (np.array([170, 140, 70], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)),
    "Yellow": (np.array([18, 50, 50], dtype=np.uint8), np.array([32, 255, 255], dtype=np.uint8)),
    "Blue":   (np.array([90, 100, 100], dtype=np.uint8), np.array([130, 255, 255], dtype=np.uint8)),
    "Green":  (np.array([40, 40, 40], dtype=np.uint8), np.array([80, 255, 255], dtype=np.uint8))
}

counters = {
    "Triangle": 0,
    "Square": 0,
    "Rectangle": 0,
    "Circle": 0,
    "Pentagon": 0,
    "Hexagon": 0
}

trackers = {
    "Red": CentroidTracker(),
    "Yellow": CentroidTracker(),
    "Blue": CentroidTracker(),
    "Green": CentroidTracker()
}

counted_objects = {
    "Red": set(),
    "Yellow": set(),
    "Blue": set(),
    "Green": set()
}

frame_width_m, frame_height_m, frame_area_m2 = calculate_frame_area()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masks = {color: cv2.inRange(hsv, lower, upper) for color, (lower, upper) in color_ranges.items()}
    masks["Red"] = cv2.bitwise_or(masks["Red1"], masks["Red2"])
    for k in ["Red1", "Red2"]:
        masks.pop(k)

    current_frame_objects = {
        "Red": [],
        "Yellow": [],
        "Blue": [],
        "Green": []
    }

    for color, mask in masks.items():
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

        edges = cv2.Canny(mask, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = []
        for cnt in contours:
            pixel_area = cv2.contourArea(cnt)
            if pixel_area < 16000:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            corners = len(approx)

            x, y, w, h = cv2.boundingRect(approx)
            rects.append((x, y, w, h))

            aspect_ratio = w / float(h)
            shape = "Unknown"
            if corners == 3:
                shape = "Triangle"
            elif corners == 4:
                shape = "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
            elif corners == 5:
                shape = "Pentagon"
            elif corners == 6:
                shape = "Hexagon"
            else:
                circularity = (4 * np.pi * pixel_area) / (peri ** 2)
                if circularity > 0.7 and 0.9 <= aspect_ratio <= 1.1:
                    shape = "Circle"

            if shape != "Unknown":
                real_area_cm2 = pixel_to_real_area(pixel_area, frame_area_m2)
                current_frame_objects[color].append((x, y, w, h, shape, approx, real_area_cm2))

        tracker = trackers[color]
        objects = tracker.update(rects)

        for (object_id, centroid) in objects.items():
            if object_id not in counted_objects[color]:
                for (x, y, w, h, shape, approx, real_area_cm2) in current_frame_objects[color]:
                    cX = int((x + x + w) / 2.0)
                    cY = int((y + y + h) / 2.0)
                    if abs(cX - centroid[0]) < 20 and abs(cY - centroid[1]) < 20:
                        counters[shape] += 1
                        counted_objects[color].add(object_id)
                        break

        for (x, y, w, h, shape, approx, real_area_cm2) in current_frame_objects[color]:
            cv2.drawContours(frame, [approx], -1, (0, 255, 255), 2)
            label = f"{color} {shape} {real_area_cm2:.2f} cm^2"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    y0 = 30
    for shape, count in counters.items():
        cv2.putText(frame, f"{shape}: {count}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y0 += 30
    
    for color, mask in masks.items():
        cv2.imshow(f"{color} Mask", mask)
    cv2.imshow("Shape Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()