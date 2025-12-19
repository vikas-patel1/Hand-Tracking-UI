import cv2
import numpy as np
import mediapipe as mp
from gestures import classify_gesture
from hand_tracker import Hand
from hud import *
from collections import deque

HEIGHT = 600
WIDTH = 800
GESTURE_HISTORY = deque(maxlen=8)
STABLE_THRESHOLD = 5

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

            gesture, value = classify_gesture(hand)

            GESTURE_HISTORY.append(gesture)

            stable_gesture = None
            if GESTURE_HISTORY.count(gesture) >= STABLE_THRESHOLD:
                stable_gesture = gesture

            if stable_gesture == "OPEN":
                draw_glow_circle(frame, palm, 120, CYAN, 3, glow=30)
                draw_glow_circle(frame, palm, 90, CYAN, 2, glow=20)
                draw_glow_circle(frame, palm, 60, ORANGE, 2, glow=10)
                draw_radial_ticks(frame, palm, 120, CYAN)
                hud_details(frame, palm, 155, CYAN)
                core_hud(frame, tuple(palm), 40)
                draw_hud_arc(frame, palm)
                cv2.putText(frame, "OPEN",(palm[0]-30, palm[1]-70),cv2.FONT_HERSHEY_SIMPLEX, 1, CYAN, 3)

            elif stable_gesture == "PINCH":
                draw_glow_circle(frame, palm, 60, ORANGE, 3, glow=20)
                cv2.putText(frame, f"PINCH {value}%",(palm[0]-60, palm[1]-70),cv2.FONT_HERSHEY_SIMPLEX, 1, ORANGE, 2)

            elif stable_gesture == "MIDDLE_FINGER":
                draw_glow_circle(frame, palm, 80, RED, 4, glow=30)
                cv2.putText(frame, "INAPPROPRIATE GESTURE",(palm[0]-140, palm[1]-100),cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 3)

            elif stable_gesture == "FIST":
                draw_glow_circle(frame, palm, 60, CYAN, 3, glow=20)
                cv2.putText(frame, "FIST",(palm[0]-30, palm[1]-70),cv2.FONT_HERSHEY_SIMPLEX, 1, ORANGE, 3)

            elif stable_gesture == "THUMBS_UP":
                draw_glow_circle(frame, palm, 70, (0, 255, 0), 3, glow=25)
                cv2.putText(frame, "THUMBS UP",(palm[0]-70, palm[1]-90),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            elif stable_gesture == "THUMBS_DOWN":
                draw_glow_circle(frame, palm, 70, (0, 0, 255), 3, glow=25)
                cv2.putText(frame, "THUMBS DOWN",(palm[0]-90, palm[1]-90),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

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
