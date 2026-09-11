"""
Record hand demonstrations via laptop camera + MediaPipe.

Each frame: 21 landmarks (x,y,z) → saved as numpy array.
Run:
    python src/capture.py --out data/demos/demo_01.npy --seconds 10
"""

import argparse
import time
import numpy as np
import cv2

try:
    import mediapipe as mp
except ImportError:
    raise ImportError("pip install mediapipe")

N_LANDMARKS = 21
LANDMARK_DIM = 3   # x, y, z (z is relative depth)


def record(out_path: str, seconds: int = 10, camera_id: int = 0):
    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    frames = []
    hands  = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    print(f"Recording for {seconds}s — press Q to stop early")
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result    = hands.process(frame_rgb)

        lm_vec = np.zeros(N_LANDMARKS * LANDMARK_DIM, np.float32)

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            for i, pt in enumerate(lm.landmark):
                lm_vec[i*3:(i+1)*3] = [pt.x, pt.y, pt.z]

        frames.append(lm_vec)

        elapsed = time.time() - t_start
        cv2.putText(frame, f"{elapsed:.1f}/{seconds}s  frames:{len(frames)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Demo recording — Q to quit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if elapsed >= seconds:
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    demo = np.array(frames, np.float32)  # (T, 63)
    np.save(out_path, demo)
    print(f"Saved {demo.shape[0]} frames → {out_path}")
    return demo


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',     required=True, help='Output .npy path')
    parser.add_argument('--seconds', type=int, default=10)
    parser.add_argument('--camera',  type=int, default=0)
    args = parser.parse_args()
    record(args.out, args.seconds, args.camera)