import cv2
import numpy as np
import mediapipe as mp

from hand_tracker import Hand
from hud import *

HEIGHT = 600
WIDTH = 800

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cam.set(cv2.CAP_PROP_FPS, 30)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hand_detector = Hand()

while True:
    ret, frame = cam.read()
    if not ret:
        break

    handData, results = hand_detector.marks(frame)

    if handData:
        for hand in handData:
            tips = []

            for idx in [4, 8, 12, 16, 20]:
                draw_glow_circle(frame, hand[idx], 10, (255, 0, 255))
                tips.append(np.array(hand[idx]))

            palm = np.array(hand[9])
            draw_glow_circle(frame, tuple(palm), 12, (0, 255, 255))

            dists = [np.linalg.norm(tip - palm) for tip in tips]
            mean_dists = np.mean(dists)

            pinch_dist = np.linalg.norm(np.array(hand[4]) - np.array(hand[8]))
            pinch_val = int(100 - min(pinch_dist, 100))

            if mean_dists > 70:
                draw_glow_circle(frame, tuple(palm), 120, CYAN, 3, glow=30)
                draw_glow_circle(frame, tuple(palm), 90, CYAN, 2, glow=20)
                draw_glow_circle(frame, tuple(palm), 60, ORANGE, 2, glow=10)
                draw_radial_ticks(frame, palm, 120, CYAN)
                hud_details(frame, palm, 155, CYAN)
                core_hud(frame, tuple(palm), 40)
                draw_hud_arc(frame, palm)

            elif pinch_val < 60:
                draw_glow_circle(frame, palm, 60, ORANGE, 3, glow=20)
                cv2.putText(frame, f'Pinch: {pinch_val}',
                            (palm[0]-40, palm[1]-70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, ORANGE, 2)
            else:
                draw_glow_circle(frame, palm, 60, CYAN, 3, glow=20)
                cv2.putText(frame, 'FIST',
                            (palm[0]-30, palm[1]-70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, ORANGE, 3)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("mywindow", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
