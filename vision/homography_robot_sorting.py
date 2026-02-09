import cv2
import cv2.aruco as aruco
import numpy as np
import time
from collections import deque
import serial
import threading
import queue

# =========================
# CONFIG
# =========================
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

XMAX = 663.0
YMAX = 316.0

REQUIRED_IDS = [0, 1, 2, 3]

# Robot arm communication
ROBOT_PORT = "COM3"  # Change to your port (e.g., "/dev/ttyUSB0" on Linux)
ROBOT_BAUD = 115200
USE_ROBOT = True  # Set to False for testing without robot

# Sorting zones (define where objects should be placed by color)
SORT_ZONES = {
    "RED": (100, 100),     # x, y in mm
    "GREEN": (100, 250),
    "BLUE": (500, 100),
    "YELLOW": (500, 250),
    "UNKNOWN": (300, 150)  # Default zone
}

tracked_objects = {}
next_object_id = 0
pick_queue = queue.Queue()  # Queue for pick commands

MAX_MATCH_DIST_MM = 80.0
COLOR_SMOOTHING = 0.8
MAX_MISSED_FRAMES = 5

STABLE_FRAMES = 8
STABLE_THRESH_MM = 5.0

# Object states
STATE_NEW = 0
STATE_STABLE = 1
STATE_QUEUED = 2      # Added: queued for picking
STATE_PICKING = 3     # Added: currently being picked
STATE_PICKED = 4      # Added: successfully picked

ANGLE_SMOOTHING = 0.7
MIN_COLOR_VOTES = 8

# Minimum time object must be stable before picking (seconds)
MIN_STABLE_TIME = 1.0

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Camera failed to open")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, 30)

# =========================
# ROBOT ARM COMMUNICATION
# =========================
class RobotArm:
    def __init__(self, port, baud, enabled=True):
        self.enabled = enabled
        self.serial = None
        self.busy = False
        self.current_pick = None
        
        if self.enabled:
            try:
                self.serial = serial.Serial(port, baud, timeout=1)
                time.sleep(2)  # Wait for Arduino to reset
                print(f"Robot arm connected on {port}")
            except Exception as e:
                print(f"Failed to connect to robot: {e}")
                self.enabled = False
    
    def send_pick_command(self, obj_id, x, y, angle, color):
        """Send PICK command to robot arm"""
        if not self.enabled or self.busy:
            return False
        
        # Get target zone for this color
        target_x, target_y = SORT_ZONES.get(color, SORT_ZONES["UNKNOWN"])
        
        # Format: PICK <id> <x_mm> <y_mm> <angle_deg> <color> <target_x> <target_y>
        command = f"PICK {obj_id} {x:.1f} {y:.1f} {angle:.1f} {color} {target_x:.1f} {target_y:.1f}\n"
        
        print(f"Sending to robot: {command.strip()}")
        
        if self.serial:
            try:
                self.serial.write(command.encode())
                self.busy = True
                self.current_pick = obj_id
                return True
            except Exception as e:
                print(f"Error sending command: {e}")
                return False
        else:
            # Simulation mode
            print(f"[SIMULATION] {command.strip()}")
            self.busy = True
            self.current_pick = obj_id
            return True
    
    def check_completion(self):
        """Check if robot has completed current pick"""
        if not self.busy:
            return None
        
        if self.serial and self.serial.in_waiting:
            try:
                response = self.serial.readline().decode().strip()
                print(f"Robot response: {response}")
                
                if "DONE" in response or "COMPLETE" in response:
                    completed_id = self.current_pick
                    self.busy = False
                    self.current_pick = None
                    return completed_id
                elif "ERROR" in response:
                    print(f"Robot error: {response}")
                    self.busy = False
                    self.current_pick = None
                    return -1  # Error code
            except Exception as e:
                print(f"Error reading response: {e}")
        
        # For simulation, auto-complete after 3 seconds
        if not self.serial and self.busy:
            # In real implementation, you'd track timing
            pass
        
        return None
    
    def is_busy(self):
        return self.busy
    
    def close(self):
        if self.serial:
            self.serial.close()

# Initialize robot arm
robot = RobotArm(ROBOT_PORT, ROBOT_BAUD, USE_ROBOT)

# =========================
# ARUCO
# =========================
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, params)

WORLD_POINTS = np.array([
    [0,     0],
    [XMAX,  0],
    [XMAX,  YMAX],
    [0,     YMAX],
], dtype=np.float32)

H = None

# =========================
# HELPERS
# =========================
def match_object(wx, wy, tracked, claimed):
    """Find closest unclaimed tracked object within MAX_MATCH_DIST_MM"""
    best_id = None
    best_dist = MAX_MATCH_DIST_MM
    for oid, obj in tracked.items():
        if oid in claimed:
            continue
        ox, oy = obj["pos"]
        d = np.hypot(wx - ox, wy - oy)
        if d < best_dist:
            best_dist = d
            best_id = oid
    return best_id

def pixel_to_world(px, py, H):
    """Transform pixel coordinates to world coordinates using homography"""
    if H is None:
        return None, None
    try:
        pt = np.array([[[px, py]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pt, H)
        return world[0][0]
    except cv2.error:
        return None, None

def is_position_stable(history):
    """Check if position has been stable for STABLE_FRAMES"""
    if len(history) < STABLE_FRAMES:
        return False
    xs = [p[0] for p in history]
    ys = [p[1] for p in history]
    return (max(xs) - min(xs) < STABLE_THRESH_MM and
            max(ys) - min(ys) < STABLE_THRESH_MM)

def detect_objects(frame, min_area=1500):
    """Detect colored objects in frame using HSV color space"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    # Optimized masking - only keep saturated, non-white pixels
    mask = cv2.inRange(s, 21, 255) & cv2.inRange(v, 0, 249)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        objects.append((cx, cy, c))
    return objects

def classify_color_hsv(frame, cnt):
    """Classify color using HSV - more robust than BGR"""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    
    # Convert to HSV for better color classification
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.mean(hsv, mask)[:3]
    
    # Filter out low saturation or dark objects
    if s < 30 or v < 50:
        return "UNKNOWN"
    
    # Hue-based classification (H is 0-180 in OpenCV)
    if h < 10 or h > 170:
        return "RED"
    elif 35 < h < 85:
        return "GREEN"
    elif 100 < h < 130:
        return "BLUE"
    elif 15 < h < 30:
        return "YELLOW"
    elif 85 < h < 100:
        return "CYAN"
    
    return "UNKNOWN"

def contour_mean_bgr(frame, cnt):
    """Calculate mean BGR color within contour"""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    b, g, r, _ = cv2.mean(frame, mask)
    return np.array([b, g, r], dtype=np.float32)

def update_color_vote(obj, frame, cnt):
    """Update color classification using voting system"""
    label = classify_color_hsv(frame, cnt)
    
    if label == "UNKNOWN":
        return

    obj["color_history"].append(label)

    # Only lock color once object is STABLE
    if obj["state"] != STATE_STABLE:
        return

    if len(obj["color_history"]) < MIN_COLOR_VOTES:
        return

    # Majority vote from recent history
    recent_votes = list(obj["color_history"])[-MIN_COLOR_VOTES:]
    counts = {}
    for c in recent_votes:
        counts[c] = counts.get(c, 0) + 1

    winner = max(counts, key=counts.get)

    # Require strong majority (70%)
    if counts[winner] >= int(0.7 * MIN_COLOR_VOTES):
        obj["color_label"] = winner

def circular_mean(angles_deg):
    """Calculate circular mean for angles (handles 0/180 wrapping)"""
    if not angles_deg:
        return 0.0
    angles_rad = np.deg2rad(angles_deg)
    sin_sum = np.mean(np.sin(angles_rad))
    cos_sum = np.mean(np.cos(angles_rad))
    mean = np.arctan2(sin_sum, cos_sum)
    return np.rad2deg(mean) % 180

def smooth_angle(current, new, alpha=ANGLE_SMOOTHING):
    """Smooth angle update considering circular nature"""
    # Find shortest path between angles
    diff = (new - current) % 180
    if diff > 90:
        diff = diff - 180
    
    # Apply smoothing to the difference
    smoothed = (current + (1 - alpha) * diff) % 180
    return smoothed

def should_pick_object(obj):
    """Determine if object is ready to be picked"""
    # Must be in STABLE state
    if obj["state"] != STATE_STABLE:
        return False
    
    # Must have identified color
    if obj["color_label"] == "UNKNOWN":
        return False
    
    # Must have locked angle
    if not obj["angle_locked"]:
        return False
    
    # Must have been stable for minimum time
    stable_duration = (time.time() - obj["stable_timestamp"]) if "stable_timestamp" in obj else 0
    if stable_duration < MIN_STABLE_TIME:
        return False
    
    return True

# =========================
# FPS
# =========================
prev = time.perf_counter()
fc = 0
fps = 0.0

# =========================
# MAIN LOOP
# =========================
print("=" * 60)
print("OBJECT SORTING SYSTEM")
print("=" * 60)
print(f"Robot arm: {'ENABLED' if robot.enabled else 'DISABLED (simulation mode)'}")
print(f"Workspace: {XMAX}mm x {YMAX}mm")
print(f"Sort zones: {SORT_ZONES}")
print("Press 'q' to quit")
print("=" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    image_pts = {}
    if ids is not None:
        ids = ids.flatten()
        aruco.drawDetectedMarkers(frame, corners, ids)
        for c, mid in zip(corners, ids):
            image_pts[mid] = c[0].mean(axis=0)

    # Establish homography if not yet done
    if H is None:
        if all(i in image_pts for i in REQUIRED_IDS):
            IMAGE_POINTS = np.array([image_pts[i] for i in REQUIRED_IDS], dtype=np.float32)
            H, _ = cv2.findHomography(IMAGE_POINTS, WORLD_POINTS)
            print("✓ Homography locked - System ready!")
        else:
            # Show which markers are missing
            missing = [i for i in REQUIRED_IDS if i not in image_pts]
            cv2.putText(frame, f"Waiting for ArUco markers: {missing}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if H is not None:
        objects = detect_objects(frame)

        claimed_ids = set()
        updated_ids = set()

        for cx, cy, cnt in objects:
            result = pixel_to_world(cx, cy, H)
            if result[0] is None:
                continue
                
            wx, wy = result
            if not (0 <= wx <= XMAX and 0 <= wy <= YMAX):
                continue

            mean_bgr = contour_mean_bgr(frame, cnt)
            oid = match_object(wx, wy, tracked_objects, claimed_ids)

            # Calculate angle from bounding box
            rect = cv2.minAreaRect(cnt)
            (_, _), (w, h), raw_angle = rect

            # OpenCV angle correction
            if w < h:
                angle = raw_angle + 90
            else:
                angle = raw_angle
            
            # Force angle into [0, 180)
            angle = angle % 180

            if oid is None:
                # New object detected
                oid = next_object_id
                next_object_id += 1
                tracked_objects[oid] = {
                    "pos": (wx, wy),
                    "color": mean_bgr,
                    "missed": 0,
                    "state": STATE_NEW,
                    "age": 1,
                    "pos_history": deque([(wx, wy)], maxlen=STABLE_FRAMES),
                    "angle": angle,
                    "color_history": deque(maxlen=MIN_COLOR_VOTES * 2),
                    "color_label": "UNKNOWN",
                    "angle_history": deque([angle], maxlen=STABLE_FRAMES),
                    "angle_locked": False,
                    "stable_timestamp": None
                }
            else:
                # Update existing object
                obj = tracked_objects[oid]
                obj["pos"] = (wx, wy)
                obj["color"] = COLOR_SMOOTHING * obj["color"] + (1 - COLOR_SMOOTHING) * mean_bgr
                obj["angle_history"].append(angle)
                
                # Update angle with smoothing
                if obj["angle_locked"]:
                    # Continue smoothing even after lock
                    obj["angle"] = smooth_angle(obj["angle"], angle)
                else:
                    # Before lock, use raw angle
                    obj["angle"] = angle
                    # Lock angle once stable
                    if obj["state"] == STATE_STABLE and len(obj["angle_history"]) == STABLE_FRAMES:
                        obj["angle"] = circular_mean(list(obj["angle_history"]))
                        obj["angle_locked"] = True
                
                obj["missed"] = 0
                obj["age"] += 1
                obj["pos_history"].append((wx, wy))
                
                # Check if position is now stable
                if obj["state"] == STATE_NEW and is_position_stable(list(obj["pos_history"])):
                    obj["state"] = STATE_STABLE
                    obj["stable_timestamp"] = time.time()
                    print(f"Object {oid} became stable at ({wx:.1f}, {wy:.1f})")
                
                # Update color classification
                update_color_vote(obj, frame, cnt)

            claimed_ids.add(oid)
            updated_ids.add(oid)

            # Draw visualization
            obj = tracked_objects[oid]
            
            # Color coding by state
            if obj["state"] == STATE_STABLE:
                state_color = (0, 255, 0)  # Green
            elif obj["state"] == STATE_QUEUED:
                state_color = (0, 165, 255)  # Orange
            elif obj["state"] == STATE_PICKING:
                state_color = (0, 0, 255)  # Red
            elif obj["state"] == STATE_PICKED:
                state_color = (128, 128, 128)  # Gray
            else:
                state_color = (0, 255, 255)  # Yellow (new)

            cv2.drawContours(frame, [cnt], -1, state_color, 2)
            
            # Draw orientation line
            theta = np.deg2rad(obj["angle"])
            length = 30
            x2 = int(cx + length * np.cos(theta))
            y2 = int(cy + length * np.sin(theta))
            cv2.line(frame, (cx, cy), (x2, y2), (255, 255, 255), 2)
            
            # Format angle text
            if obj["angle_locked"]:
                angle_text = f"{int(obj['angle'])}°"
            else:
                angle_text = "??"
            
            label = f"ID{oid} {obj['color_label']} {angle_text}"
            
            # Draw label
            text_color = (0, 255, 0) if obj["color_label"] != "UNKNOWN" else (0, 255, 255)
            cv2.putText(frame, label, (cx + 8, cy - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            # Draw position in mm
            pos_label = f"({wx:.0f},{wy:.0f})"
            cv2.putText(frame, pos_label, (cx + 8, cy + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Remove objects that haven't been seen recently
        for oid in list(tracked_objects.keys()):
            if oid not in updated_ids:
                obj = tracked_objects[oid]
                obj["missed"] += 1
                
                # Don't delete objects being picked or queued
                if obj["state"] in [STATE_QUEUED, STATE_PICKING]:
                    continue
                    
                if obj["missed"] > MAX_MISSED_FRAMES:
                    print(f"Lost object {oid}")
                    del tracked_objects[oid]
        
        # PICKING LOGIC
        # Check for completed picks
        completed = robot.check_completion()
        if completed is not None:
            if completed > 0 and completed in tracked_objects:
                print(f"✓ Object {completed} picked successfully")
                tracked_objects[completed]["state"] = STATE_PICKED
                # Remove from tracking after short delay
                del tracked_objects[completed]
            elif completed == -1:
                print("✗ Pick failed - robot error")
        
        # Queue new objects for picking
        if not robot.is_busy():
            # Find objects ready to pick
            ready_objects = []
            for oid, obj in tracked_objects.items():
                if should_pick_object(obj):
                    ready_objects.append((oid, obj))
            
            # Pick oldest stable object first
            if ready_objects:
                ready_objects.sort(key=lambda x: x[1]["stable_timestamp"])
                oid, obj = ready_objects[0]
                
                # Send pick command
                x, y = obj["pos"]
                success = robot.send_pick_command(
                    oid, x, y, obj["angle"], obj["color_label"]
                )
                
                if success:
                    obj["state"] = STATE_QUEUED
                    print(f"→ Queued object {oid} for picking")

    # Calculate and display FPS
    fc += 1
    now = time.perf_counter()
    if now - prev >= 1.0:
        fps = fc / (now - prev)
        fc = 0
        prev = now

    # Status overlay
    cv2.putText(frame, f"FPS {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.putText(frame, f"Objects: {len(tracked_objects)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    robot_status = "PICKING" if robot.is_busy() else "READY"
    status_color = (0, 165, 255) if robot.is_busy() else (0, 255, 0)
    cv2.putText(frame, f"Robot: {robot_status}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Reset homography
        H = None
        print("Homography reset - reposition ArUco markers")

cap.release()
cv2.destroyAllWindows()
robot.close()
print("\nSystem stopped")
