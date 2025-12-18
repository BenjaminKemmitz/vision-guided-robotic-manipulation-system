import cv2
import time
import os

# Configuration
CAMERA_INDEX = 0
FRAME_WIDTH =1980
FRAME_HEIGHT = 1080
TARGET_FPS = 30
SAVE_DIR = "../images/captured"

os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize Camera
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

# FPS Tracking
prev_time = time.perf_counter()
frame_count = 0
fps = 0.0

# Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("WARNING: Frame capture failed")
        continue

    frame_count += 1
    current_time = time.perf_counter()
    elapsed = current_time - prev_time

    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        prev_time = current_time

    # Overlay info
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"{frame.shape[1]}x{frame.shape[0]}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Camera Feed", frame)

    key = cv2.waitKey(1) & 0xFF

    # Save frame
    if key == ord('s'):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{SAVE_DIR}/frame_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

    # Quit
    if key == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Camera released. Exiting.")
