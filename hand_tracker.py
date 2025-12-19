import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands


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

        return allHands, results
