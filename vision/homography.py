import cv2
import cv2.aruco as aruco
import numpy as np
import time

# =========================
# CONFIGURATION
# =========================
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

XMAX = 663.0   # mm
YMAX = 316.0  # mm

REQUIRED_IDS = [0, 1, 2, 3]

# =========================
# CAMERA SETUP
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

# Disable autofocus (IMPORTANT)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, 30)

# =========================
# ARUCO SETUP
# =========================
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, params)

# =========================
# WORLD POINTS (mm)
# =========================
WORLD_POINTS = np.array([
    [0,     0],      # ID 0
    [XMAX,  0],      # ID 1
    [XMAX,  YMAX],   # ID 2
    [0,     YMAX],   # ID 3
], dtype=np.float32)

H = None  # homography matrix

# =========================
# FPS TRACKING
# =========================
prev_time = time.perf_counter()
frame_count = 0
fps = 0.0

# =========================
# MOUSE CALLBACK
# =========================
def mouse_callback(event, x, y, flags, param):
    global H
    if event == cv2.EVENT_MOUSEMOVE and H is not None:
        pt = np.array([[[x, y]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pt, H)
        wx, wy = world[0][0]
        print(f"X = {wx:.1f} mm, Y = {wy:.1f} mm")

cv2.namedWindow("Camera Feed")
cv2.setMouseCallback("Camera Feed", mouse_callback)

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    image_points = {}

    if ids is not None:
        ids = ids.flatten()
        aruco.drawDetectedMarkers(frame, corners, ids)

        for corner, marker_id in zip(corners, ids):
            center = corner[0].mean(axis=0)
            image_points[marker_id] = center

            cx, cy = int(center[0]), int(center[1])
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"ID {marker_id}",
                        (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1)

    # Compute homography ONCE
    if H is None and all(i in image_points for i in REQUIRED_IDS):
        IMAGE_POINTS = np.array([
            image_points[0],
            image_points[1],
            image_points[2],
            image_points[3],
        ], dtype=np.float32)

        H, _ = cv2.findHomography(IMAGE_POINTS, WORLD_POINTS)
        print("Homography computed!")

    # FPS
    frame_count += 1
    now = time.perf_counter()
    if now - prev_time >= 1.0:
        fps = frame_count / (now - prev_time)
        frame_count = 0
        prev_time = now

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    if H is None:
        cv2.putText(frame,
                    "Move camera until all 4 markers are visible",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
    else:
        cv2.putText(frame,
                    "Homography LOCKED",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
