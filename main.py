import cv2
import mediapipe as mp
# import numpy as np

HEIGHT = 600
WIDTH = 800

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cam.set(cv2.CAP_PROP_FPS, 30)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


class Hand:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def marks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        allHands = []
        height, width, _ = frame.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                myhand = []
                for lm in hand_landmarks.landmark:
                    cx = int(lm.x * width)
                    cy = int(lm.y * height)
                    myhand.append((cx, cy))
                allHands.append(myhand)

        return allHands


hand_detector = Hand()

while True:
    ret, frame = cam.read()
    if not ret:
        print("Camera couldn't open")
        break
    
    handData = hand_detector.marks(frame)
    
    if handData:
        for hand in handData:
            # fingertip indices
            for idx in [4, 8, 12, 16, 20]:
                cv2.circle(frame, hand[idx], 10, (255, 0, 255), -1)

            # palm center (middle finger MCP)
            palm = hand[9]
            cv2.circle(frame, palm, 12, (0, 255, 255), -1)

    cv2.imshow("mywindow", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
