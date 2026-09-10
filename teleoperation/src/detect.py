"""
Hand Landmark Detection for Teleoperation.

Captures hand landmarks from webcam using MediaPipe.
Returns normalized 21 keypoints (x, y, z) per hand.
"""

import cv2
import mediapipe as mp
import numpy as np

# ── MediaPipe setup ───────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def get_landmarks(hand_landmarks) -> np.ndarray:
    """
    Extract 21 landmarks as numpy array (21, 3).
    Values are normalized [0, 1] relative to image size.
    """
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32
    )


def run_detection(callback):
    """
    Opens webcam, detects hand landmarks each frame,
    calls callback(landmarks) with shape (21, 3).

    Press Q to quit.
    """
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )
                landmarks = get_landmarks(hand)
                callback(landmarks)

            cv2.imshow("Teleoperation — Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    def print_landmarks(lm):
        print(f"Wrist: {lm[0].round(3)}  Index tip: {lm[8].round(3)}")

    run_detection(print_landmarks)