import cv2
import cv2.aruco as aruco
import numpy as np
import time
import os

# =========================
# CONFIGURATION
# =========================

CAMERA_INDEX = 1
FRAME_WIDTH = 1980
FRAME_HEIGHT = 1080
TARGET_FPS = 30
SAVE_DIR = "../images/captured"

X_MAX = 66.3   # mm
Y_MAX = 31.6   # mm
S = 3.9        # mm (marker size, for reference)

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# CAMERA INIT
# =========================

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("ERROR: Could not open camera.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

print("Camera initialized")
print(f"Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
print("Controls:")
print("  s = save frame")
print("  q = quit")

# =========================
# ARUCO SETUP
# =========================

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
aruco_params = aruco.DetectorParameters()
aruco_detector = aruco.ArucoDetector(aruco_dict, aruco_params)

# Marker world coordinates (mm)
MARKER_WORLD = {
    0: (0.0,    0.0),
    1: (X_MAX,  0.0),
    2: (0.0,    Y_MAX),
    3: (X_MAX,  Y_MAX),
}

H = None

# =========================
# MOUSE TRACKING
# =========================

mouse_u, mouse_v = None, None

def mouse_callback(event, x, y, flags, param):
    global mouse_u, mouse_v
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_u, mouse_v = x, y

cv2.namedWindow("Camera Feed")
cv2.setMouseCallback("Camera Feed", mouse_callback)

# =========================
# UTILITY
# =========================

def pixel_to_table(u, v, H):
    p = np.array([u, v, 1.0])
    P = H @ p
    P /= P[2]
    return P[0], P[1]

# =========================
# FPS TRACKING
# =========================

prev_time = time.perf_counter()
frame_count = 0
fps = 0.0

# =========================
# MAIN LOOP
# =========================

while True:
    ret, frame = cap.read()
    if not ret:
        print("WARNING: Frame capture failed")
        continue

    # FPS
    frame_count += 1
    current_time = time.perf_counter()
    elapsed = current_time - prev_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        prev_time = current_time

    # =========================
    # ARUCO + HOMOGRAPHY
    # =========================

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco_detector.detectMarkers(gray)

    img_pts = []
    world_pts = []

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in MARKER_WORLD:
                c = corners[i][0]          # 4x2
                center = c.mean(axis=0)

                img_pts.append(center)
                world_pts.append(MARKER_WORLD[marker_id])

                cv2.polylines(frame, [c.astype(int)], True, (0,255,0), 2)
                cv2.circle(frame, tuple(center.astype(int)), 4, (0,255,0), -1)
                cv2.putText(
                    frame, f"ID {marker_id}",
                    tuple(center.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,0), 2
                )

        if len(img_pts) >= 4:
            H, _ = cv2.findHomography(
                np.array(img_pts, dtype=np.float32),
                np.array(world_pts, dtype=np.float32)
            )

    # =========================
    # OVERLAYS
    # =========================

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.putText(frame, f"{frame.shape[1]}x{frame.shape[0]}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    if H is not None and mouse_u is not None:
        X, Y = pixel_to_table(mouse_u, mouse_v, H)

        cv2.circle(frame, (mouse_u, mouse_v), 4, (0,0,255), -1)
        cv2.putText(
            frame,
            f"X={X:.2f} mm  Y={Y:.2f} mm",
            (mouse_u + 10, mouse_v - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,255),
            2
        )

    cv2.imshow("Camera Feed", frame)

    # =========================
    # KEYS
    # =========================

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{SAVE_DIR}/frame_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

    if key == ord('q'):
        break

# =========================
# CLEANUP
# =========================

cap.release()
cv2.destroyAllWindows()
print("Camera released. Exiting.")

