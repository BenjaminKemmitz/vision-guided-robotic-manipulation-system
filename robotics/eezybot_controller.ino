/*
 * EEZYbotARM MK2 - Object Sorting Controller
 * 
 * This Arduino sketch receives PICK commands from the vision system
 * and controls the robotic arm to pick and sort objects.
 * 
 * Command format: PICK <id> <x_mm> <y_mm> <angle_deg> <color> <target_x> <target_y>
 * Response: DONE <id> or ERROR <message>
 * 
 * Hardware connections:
 * - Servo 1 (Base rotation): Pin 9
 * - Servo 2 (Shoulder): Pin 10
 * - Servo 3 (Elbow): Pin 11
 * - Servo 4 (Gripper): Pin 6
 */

#include <Servo.h>

// Servo objects
Servo servoBase;     // Base rotation (0-180°)
Servo servoShoulder; // Shoulder joint
Servo servoElbow;    // Elbow joint
Servo servoGripper;  // Gripper open/close

// Pin assignments
const int PIN_BASE = 9;
const int PIN_SHOULDER = 10;
const int PIN_ELBOW = 11;
const int PIN_GRIPPER = 6;

// Servo angle limits (adjust for your specific arm)
const int BASE_MIN = 0;
const int BASE_MAX = 180;
const int SHOULDER_MIN = 20;
const int SHOULDER_MAX = 160;
const int ELBOW_MIN = 20;
const int ELBOW_MAX = 160;
const int GRIPPER_OPEN = 90;   // Gripper open position
const int GRIPPER_CLOSE = 40;  // Gripper closed position

// Arm dimensions (in mm) - ADJUST THESE FOR YOUR ARM
const float ARM_BASE_HEIGHT = 50.0;  // Height of base to shoulder joint
const float ARM_SHOULDER_LENGTH = 135.0;  // Shoulder to elbow length
const float ARM_ELBOW_LENGTH = 147.0;     // Elbow to gripper length
const float GRIPPER_OFFSET = 50.0;        // Gripper tip offset

// Movement speed
const int MOVE_DELAY = 15;  // ms between servo steps (lower = faster)

// Home position
const int HOME_BASE = 90;
const int HOME_SHOULDER = 90;
const int HOME_ELBOW = 90;

// Current position tracking
int currentBase = HOME_BASE;
int currentShoulder = HOME_SHOULDER;
int currentElbow = HOME_ELBOW;
int currentGripper = GRIPPER_OPEN;

// Command buffer
String inputBuffer = "";

void setup() {
  Serial.begin(115200);
  
  // Attach servos
  servoBase.attach(PIN_BASE);
  servoShoulder.attach(PIN_SHOULDER);
  servoElbow.attach(PIN_ELBOW);
  servoGripper.attach(PIN_GRIPPER);
  
  // Move to home position
  moveToHome();
  
  Serial.println("EEZYbotARM Ready");
}

void loop() {
  // Check for incoming commands
  while (Serial.available() > 0) {
    char c = Serial.read();
    
    if (c == '\n') {
      // Process complete command
      processCommand(inputBuffer);
      inputBuffer = "";
    } else {
      inputBuffer += c;
    }
  }
}

void processCommand(String cmd) {
  cmd.trim();
  
  if (cmd.startsWith("PICK")) {
    // Parse: PICK <id> <x_mm> <y_mm> <angle_deg> <color> <target_x> <target_y>
    int id, angle;
    float x, y, target_x, target_y;
    String color;
    
    // Simple parsing
    int pos = 5; // Skip "PICK "
    int nextSpace;
    
    // Parse ID
    nextSpace = cmd.indexOf(' ', pos);
    id = cmd.substring(pos, nextSpace).toInt();
    pos = nextSpace + 1;
    
    // Parse X
    nextSpace = cmd.indexOf(' ', pos);
    x = cmd.substring(pos, nextSpace).toFloat();
    pos = nextSpace + 1;
    
    // Parse Y
    nextSpace = cmd.indexOf(' ', pos);
    y = cmd.substring(pos, nextSpace).toFloat();
    pos = nextSpace + 1;
    
    // Parse angle
    nextSpace = cmd.indexOf(' ', pos);
    angle = cmd.substring(pos, nextSpace).toInt();
    pos = nextSpace + 1;
    
    // Parse color
    nextSpace = cmd.indexOf(' ', pos);
    color = cmd.substring(pos, nextSpace);
    pos = nextSpace + 1;
    
    // Parse target X
    nextSpace = cmd.indexOf(' ', pos);
    target_x = cmd.substring(pos, nextSpace).toFloat();
    pos = nextSpace + 1;
    
    // Parse target Y
    target_y = cmd.substring(pos).toFloat();
    
    // Execute pick and place
    bool success = pickAndPlace(x, y, angle, target_x, target_y);
    
    if (success) {
      Serial.print("DONE ");
      Serial.println(id);
    } else {
      Serial.print("ERROR ");
      Serial.println(id);
    }
    
  } else if (cmd == "HOME") {
    moveToHome();
    Serial.println("DONE HOME");
    
  } else if (cmd == "STATUS") {
    Serial.print("STATUS Base:");
    Serial.print(currentBase);
    Serial.print(" Shoulder:");
    Serial.print(currentShoulder);
    Serial.print(" Elbow:");
    Serial.println(currentElbow);
    
  } else {
    Serial.println("ERROR Unknown command");
  }
}

bool pickAndPlace(float x, float y, int objAngle, float targetX, float targetY) {
  Serial.print("Picking at (");
  Serial.print(x);
  Serial.print(", ");
  Serial.print(y);
  Serial.println(")");
  
  // 1. Open gripper
  moveGripper(GRIPPER_OPEN);
  delay(300);
  
  // 2. Move to pick position (above object)
  if (!moveToPosition(x, y, ARM_BASE_HEIGHT + 50)) {
    Serial.println("ERROR: Cannot reach pick position");
    return false;
  }
  delay(300);
  
  // 3. Lower to grasp height
  if (!moveToPosition(x, y, 10)) {  // Lower to 10mm above table
    Serial.println("ERROR: Cannot reach grasp position");
    return false;
  }
  delay(300);
  
  // 4. Close gripper
  moveGripper(GRIPPER_CLOSE);
  delay(500);  // Wait for grip
  
  // 5. Lift object
  if (!moveToPosition(x, y, ARM_BASE_HEIGHT + 80)) {
    Serial.println("ERROR: Cannot lift object");
    return false;
  }
  delay(300);
  
  // 6. Move to target position (above)
  if (!moveToPosition(targetX, targetY, ARM_BASE_HEIGHT + 80)) {
    Serial.println("ERROR: Cannot reach target");
    return false;
  }
  delay(300);
  
  // 7. Lower to place height
  if (!moveToPosition(targetX, targetY, 20)) {
    Serial.println("ERROR: Cannot reach place position");
    return false;
  }
  delay(300);
  
  // 8. Open gripper
  moveGripper(GRIPPER_OPEN);
  delay(500);
  
  // 9. Return to home
  moveToHome();
  
  return true;
}

bool moveToPosition(float x, float y, float z) {
  // Calculate inverse kinematics
  int baseAngle, shoulderAngle, elbowAngle;
  
  if (!inverseKinematics(x, y, z, baseAngle, shoulderAngle, elbowAngle)) {
    return false;
  }
  
  // Constrain angles to safe limits
  baseAngle = constrain(baseAngle, BASE_MIN, BASE_MAX);
  shoulderAngle = constrain(shoulderAngle, SHOULDER_MIN, SHOULDER_MAX);
  elbowAngle = constrain(elbowAngle, ELBOW_MIN, ELBOW_MAX);
  
  // Move servos smoothly
  moveServoSmooth(servoBase, currentBase, baseAngle);
  currentBase = baseAngle;
  
  moveServoSmooth(servoShoulder, currentShoulder, shoulderAngle);
  currentShoulder = shoulderAngle;
  
  moveServoSmooth(servoElbow, currentElbow, elbowAngle);
  currentElbow = elbowAngle;
  
  return true;
}

bool inverseKinematics(float x, float y, float z, int &base, int &shoulder, int &elbow) {
  // Calculate base rotation angle
  base = (int)(atan2(y, x) * 180.0 / PI);
  base = 90 + base;  // Adjust for servo orientation
  
  // Calculate reach distance in XY plane
  float reach = sqrt(x * x + y * y);
  
  // Adjust Z for base height
  float z_adj = z - ARM_BASE_HEIGHT;
  
  // Calculate distance to target
  float distance = sqrt(reach * reach + z_adj * z_adj);
  
  // Check if target is reachable
  float maxReach = ARM_SHOULDER_LENGTH + ARM_ELBOW_LENGTH;
  float minReach = abs(ARM_SHOULDER_LENGTH - ARM_ELBOW_LENGTH);
  
  if (distance > maxReach || distance < minReach) {
    return false;  // Target unreachable
  }
  
  // Calculate elbow angle using law of cosines
  float cosElbow = (ARM_SHOULDER_LENGTH * ARM_SHOULDER_LENGTH + 
                    ARM_ELBOW_LENGTH * ARM_ELBOW_LENGTH - 
                    distance * distance) / 
                   (2.0 * ARM_SHOULDER_LENGTH * ARM_ELBOW_LENGTH);
  
  if (cosElbow < -1.0 || cosElbow > 1.0) {
    return false;
  }
  
  float elbowAngleRad = acos(cosElbow);
  elbow = (int)(elbowAngleRad * 180.0 / PI);
  
  // Calculate shoulder angle
  float angle1 = atan2(z_adj, reach);
  float angle2 = acos((ARM_SHOULDER_LENGTH * ARM_SHOULDER_LENGTH + 
                       distance * distance - 
                       ARM_ELBOW_LENGTH * ARM_ELBOW_LENGTH) / 
                      (2.0 * ARM_SHOULDER_LENGTH * distance));
  
  float shoulderAngleRad = angle1 + angle2;
  shoulder = (int)(shoulderAngleRad * 180.0 / PI);
  
  return true;
}

void moveServoSmooth(Servo &servo, int fromAngle, int toAngle) {
  if (fromAngle < toAngle) {
    for (int pos = fromAngle; pos <= toAngle; pos++) {
      servo.write(pos);
      delay(MOVE_DELAY);
    }
  } else {
    for (int pos = fromAngle; pos >= toAngle; pos--) {
      servo.write(pos);
      delay(MOVE_DELAY);
    }
  }
}

void moveGripper(int angle) {
  moveServoSmooth(servoGripper, currentGripper, angle);
  currentGripper = angle;
}

void moveToHome() {
  Serial.println("Moving to home position");
  
  moveServoSmooth(servoGripper, currentGripper, GRIPPER_OPEN);
  currentGripper = GRIPPER_OPEN;
  
  moveServoSmooth(servoElbow, currentElbow, HOME_ELBOW);
  currentElbow = HOME_ELBOW;
  
  moveServoSmooth(servoShoulder, currentShoulder, HOME_SHOULDER);
  currentShoulder = HOME_SHOULDER;
  
  moveServoSmooth(servoBase, currentBase, HOME_BASE);
  currentBase = HOME_BASE;
}
