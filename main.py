import cv2
import mediapipe as mp
import numpy as np

HEIGHT = 600
WIDTH = 800

# Colors for UI
CYAN = (255, 255, 0)
ORANGE = (0, 180, 255)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
CORE = (0, 255, 180)

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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

        return allHands, results


hand_detector = Hand()


def draw_glow_circle(frame, center, radius, color, thickness=2, glow=15):
    for g in range(glow, 0, -3):
        alpha = 0.08 + 0.12 * (g / glow)
        overlay = frame.copy()
        cv2.circle(overlay, center, radius + g, color, thickness)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.circle(frame, center, radius, color, thickness)

def draw_radial_ticks(img,center,radius,color,num_ticks=24,length=22):
    for i in range(num_ticks):
        angle=np.deg2rad(i*(360/num_ticks))
        x1=int(center[0]+(radius-length)*np.cos(angle))
        y1=int(center[1]+(radius-length)*np.sin(angle))
        x2=int(center[0]+(radius)*np.cos(angle))
        y2=int(center[1]+(radius)*np.sin(angle))

        cv2.line(img,(x1,y1),(x2,y2),color,3)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Camera couldn't open")
        break

    handData, results = hand_detector.marks(frame)

    if handData:
        for hand in handData:
            tips = []  # MUST be outside fingertip loop

            for idx in [4, 8, 12, 16, 20]:
                draw_glow_circle(frame, hand[idx], 10, (255, 0, 255))
                tips.append(np.array(hand[idx]))

            palm = np.array(hand[9])
            draw_glow_circle(frame, tuple(palm), 12, (0, 255, 255))

            # Distance calculation (CORRECT)
            dists = [np.linalg.norm(tip - palm) for tip in tips]
            mean_dists = np.mean(dists)

            if mean_dists > 70:
                draw_glow_circle(frame, tuple(palm), 120, CYAN, 3, glow=30)
                draw_glow_circle(frame, tuple(palm), 90, CYAN, 2, glow=20)
                draw_glow_circle(frame, tuple(palm), 60, ORANGE, 2, glow=10)
                draw_radial_ticks(frame,palm,120,CYAN,num_ticks=24,length=22)
                
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
