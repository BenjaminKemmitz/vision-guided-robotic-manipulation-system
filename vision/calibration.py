import cv2
import cv2.aruco as aruco
import numpy as np
import time
import os

# =========================
# CONFIGURATION
# =========================

CAMERA_INDEX = 1
FRAME_WIDTH = 1280        # lowered for speed & sharpness
FRAME_HEIGHT = 720
TARGET_FPS = 30

SAVE_DIR = "../images/captured"
os.makedirs(SAVE_DIR, exist_ok=True)

# Table dimensions (mm)
X_MAX = 663.0
Y_MAX = 316.0
S = 39.0  # marker size (mm, informational)

# ArUco performance tuning
DETECT_SCALE = 0.5        # downscale factor for detection
DETECT_EVERY = 5          # detect every N frames

# =========================
# CAMERA INIT
# =========================

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("ERROR: Could not open camera")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

# Disable auto junk (backend dependent)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, 0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)

# Warm up camera (CRITICAL)
for _ in range(60):
    cap.read()
    time.sleep(0.01)

print("Camera initialized")
print("Actual resolution:",
      int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# =========================
# ARUCO SETUP
# =========================

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
aruco_params = aruco.DetectorParameters()
aruco_detector = aruco.ArucoDetector(aruco_dict, aruco_params)

MARKER_WORLD = {
    0: (0.0,    0.0),
    1: (X_MAX,  0.0),
    2: (0.0,    Y_MAX),
    3: (X_MAX,  Y_MAX),
}

H = None
H_LOCKED = False

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
frame_id = 0

# =========================
# MAIN LOOP
# =========================

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_id += 1

    # FPS calc
    frame_count += 1
    now = time.perf_counter()
    if now - prev_time >= 1.0:
        fps = frame_count / (now - prev_time)
        frame_count = 0
        prev_time = now

    # =========================
    # ARUCO DETECTION (OPTIMIZED)
    # =========================

    if not H_LOCKED and frame_id % DETECT_EVERY == 0:
        small = cv2.resize(frame, None, fx=DETECT_SCALE, fy=DETECT_SCALE)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = aruco_detector.detectMarkers(gray)

        img_pts = []
        world_pts = []

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in MARKER_WORLD:
                    c = corners[i][0] / DETECT_SCALE
                    center = c.mean(axis=0)

                    img_pts.append(center)
                    world_pts.append(MARKER_WORLD[marker_id])

                    cv2.polylines(frame, [c.astype(int)], True, (0,255,0), 2)
                    cv2.putText(
                        frame, f"ID {marker_id}",
                        tuple(center.astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2
                    )

            if len(img_pts) >= 4:
                H, _ = cv2.findHomography(
                    np.array(img_pts, np.float32),
                    np.array(world_pts, np.float32)
                )
                H_LOCKED = True
                print("Homography locked")

    # =========================
    # OVERLAYS
    # =========================

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    if H is not None and mouse_u is not None:
        X, Y = pixel_to_table(mouse_u, mouse_v, H)
        cv2.circle(frame, (mouse_u, mouse_v), 4, (0,0,255), -1)
        cv2.putText(
            frame,
            f"X={X:.1f} mm  Y={Y:.1f} mm",
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
        ts = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"{SAVE_DIR}/frame_{ts}.jpg", frame)

    if key == ord('r'):
        H_LOCKED = False
        H = None
        print("Homography reset")

    if key == ord('q'):
        break

# =========================
# CLEANUP
# =========================

cap.release()
cv2.destroyAllWindows()
print("Exited cleanly")
