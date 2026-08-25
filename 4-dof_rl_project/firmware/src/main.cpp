/**
 * ESP32-S3 Robot Arm — Serial Control Firmware
 *
 * Receives joint angles from Python deploy script via USB Serial.
 * Protocol: "<base>,<shoulder>,<elbow>,<gripper>\n"
 * Example:  "90,75,110,90\n"
 *
 * Responds with "OK\n" after moving servos.
 *
 * Joints:
 *   Servo 0 — Base     (GPIO 4)   0–180°
 *   Servo 1 — Shoulder (GPIO 5)  30–150°
 *   Servo 2 — Elbow    (GPIO 6)  30–150°
 *   Servo 3 — Gripper  (GPIO 7)  60–120°
 */

#include <Arduino.h>
#include <ESP32Servo.h>

// ── Pin assignments ───────────────────────────────────────────
static const int PIN_BASE     = 4;
static const int PIN_SHOULDER = 5;
static const int PIN_ELBOW    = 6;
static const int PIN_GRIPPER  = 7;

// ── Joint limits (degrees) ────────────────────────────────────
static const int BASE_MIN     =   0,  BASE_MAX     = 180;
static const int SHOULDER_MIN =  30,  SHOULDER_MAX = 150;
static const int ELBOW_MIN    =  30,  ELBOW_MAX    = 150;
static const int GRIPPER_MIN  =  60,  GRIPPER_MAX  = 120;

// ── Home position ─────────────────────────────────────────────
static const int HOME_BASE     = 90;
static const int HOME_SHOULDER = 90;
static const int HOME_ELBOW    = 90;
static const int HOME_GRIPPER  = 90;

// ── Serial ────────────────────────────────────────────────────
static const int BAUD_RATE    = 115200;
static const int MAX_MSG_LEN  = 32;

// ── Servo objects ─────────────────────────────────────────────
Servo servo_base;
Servo servo_shoulder;
Servo servo_elbow;
Servo servo_gripper;


// ── Helpers ───────────────────────────────────────────────────

int clamp(int value, int min_val, int max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

void move_to(int base, int shoulder, int elbow, int gripper) {
    servo_base.write(    clamp(base,     BASE_MIN,     BASE_MAX));
    servo_shoulder.write(clamp(shoulder, SHOULDER_MIN, SHOULDER_MAX));
    servo_elbow.write(   clamp(elbow,    ELBOW_MIN,    ELBOW_MAX));
    servo_gripper.write( clamp(gripper,  GRIPPER_MIN,  GRIPPER_MAX));
}

void detach_all() {
    servo_base.detach();
    servo_shoulder.detach();
    servo_elbow.detach();
    servo_gripper.detach();
}

void attach_all() {
    servo_base.attach(PIN_BASE);
    servo_shoulder.attach(PIN_SHOULDER);
    servo_elbow.attach(PIN_ELBOW);
    servo_gripper.attach(PIN_GRIPPER);
}


// ── Parse incoming message ─────────────────────────────────────

/**
 * Parse a message of the form "90,75,110,90" into four integers.
 *
 * Returns true on success, false if the format is invalid.
 */
bool parse_angles(const char* msg, int& base, int& shoulder, int& elbow, int& gripper) {
    int parsed = sscanf(msg, "%d,%d,%d,%d", &base, &shoulder, &elbow, &gripper);
    return parsed == 4;
}


// ── Setup ─────────────────────────────────────────────────────

void setup() {
    Serial.begin(BAUD_RATE);
    delay(500);

    attach_all();

    // Move to home position on startup
    move_to(HOME_BASE, HOME_SHOULDER, HOME_ELBOW, HOME_GRIPPER);
    delay(800);
    detach_all();   // reduce jitter when idle

    Serial.println("READY");
}


// ── Main loop ─────────────────────────────────────────────────

void loop() {
    static char  buf[MAX_MSG_LEN];
    static int   buf_idx = 0;

    // Read bytes until newline
    while (Serial.available()) {
        char c = (char)Serial.read();

        if (c == '\n' || c == '\r') {
            if (buf_idx > 0) {
                buf[buf_idx] = '\0';   // null-terminate

                int base, shoulder, elbow, gripper;
                if (parse_angles(buf, base, shoulder, elbow, gripper)) {
                    attach_all();
                    move_to(base, shoulder, elbow, gripper);
                    delay(300);        // wait for servos to reach position
                    detach_all();
                    Serial.println("OK");
                } else {
                    Serial.print("ERR: bad format: ");
                    Serial.println(buf);
                }

                buf_idx = 0;   // reset buffer
            }
        } else if (buf_idx < MAX_MSG_LEN - 1) {
            buf[buf_idx++] = c;
        }
    }
}