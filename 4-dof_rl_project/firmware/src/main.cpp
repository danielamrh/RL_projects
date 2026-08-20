// ============================================================
//  Robot Arm - Full Test with detach (no jitter)
//  Board:   ESP32-S3-DevKitC-1
//  Servos:  4x SG90 on GPIO 4, 5, 6, 7
//  Home:    all at 90°
// ============================================================

#include <Arduino.h>
#include <ESP32Servo.h>

const int PIN_BASE     = 4;
const int PIN_SHOULDER = 5;
const int PIN_ELBOW    = 6;
const int PIN_GRIPPER  = 7;

const int HOME_BASE     = 90;
const int HOME_SHOULDER = 90;
const int HOME_ELBOW    = 90;
const int HOME_GRIPPER  = 90;

Servo base;
Servo shoulder;
Servo elbow;
Servo gripper;

void attachAll() {
  base.attach(PIN_BASE,         500, 2400);
  shoulder.attach(PIN_SHOULDER, 500, 2400);
  elbow.attach(PIN_ELBOW,       500, 2400);
  gripper.attach(PIN_GRIPPER,   500, 2400);
}

void detachAll() {
  base.detach();
  shoulder.detach();
  elbow.detach();
  gripper.detach();
}

void moveSmooth(Servo& servo, int pin, int fromDeg, int toDeg, int speed = 20) {
  servo.attach(pin, 500, 2400);
  if (fromDeg < toDeg) {
    for (int pos = fromDeg; pos <= toDeg; pos++) {
      servo.write(pos);
      delay(speed);
    }
  } else {
    for (int pos = fromDeg; pos >= toDeg; pos--) {
      servo.write(pos);
      delay(speed);
    }
  }
  delay(300);
  servo.detach();  // stop PWM → no jitter
}

void goHome() {
  moveSmooth(gripper,  PIN_GRIPPER,  gripper.read(),  HOME_GRIPPER);
  moveSmooth(elbow,    PIN_ELBOW,    elbow.read(),    HOME_ELBOW);
  moveSmooth(shoulder, PIN_SHOULDER, shoulder.read(), HOME_SHOULDER);
  moveSmooth(base,     PIN_BASE,     base.read(),     HOME_BASE);
}

void setup() {
  base.setPeriodHertz(50);
  shoulder.setPeriodHertz(50);
  elbow.setPeriodHertz(50);
  gripper.setPeriodHertz(50);

  // Move to home and detach
  attachAll();
  base.write(HOME_BASE);
  shoulder.write(HOME_SHOULDER);
  elbow.write(HOME_ELBOW);
  gripper.write(HOME_GRIPPER);
  delay(1000);
  detachAll();
}

void loop() {
  // Base sweep
  moveSmooth(base, PIN_BASE, 90, 45);
  delay(500);
  moveSmooth(base, PIN_BASE, 45, 135);
  delay(500);
  moveSmooth(base, PIN_BASE, 135, 90);
  delay(500);

  // Shoulder sweep
  moveSmooth(shoulder, PIN_SHOULDER, 90, 45);
  delay(500);
  moveSmooth(shoulder, PIN_SHOULDER, 45, 135);
  delay(500);
  moveSmooth(shoulder, PIN_SHOULDER, 135, 90);
  delay(500);

  // Elbow sweep
  moveSmooth(elbow, PIN_ELBOW, 90, 45);
  delay(500);
  moveSmooth(elbow, PIN_ELBOW, 45, 135);
  delay(500);
  moveSmooth(elbow, PIN_ELBOW, 135, 90);
  delay(500);

  // Gripper open/close
  moveSmooth(gripper, PIN_GRIPPER, 90, 60);
  delay(500);
  moveSmooth(gripper, PIN_GRIPPER, 60, 120);
  delay(500);
  moveSmooth(gripper, PIN_GRIPPER, 120, 90);
  delay(500);

  delay(3000);
}