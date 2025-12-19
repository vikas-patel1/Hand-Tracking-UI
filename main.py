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

#drawing glowing outer circle
def draw_glow_circle(frame, center, radius, color, thickness=2, glow=15):
    for g in range(glow, 0, -3):
        alpha = 0.08 + 0.12 * (g / glow)
        overlay = frame.copy()
        cv2.circle(overlay, center, radius + g, color, thickness)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.circle(frame, center, radius, color, thickness)

#drawing inner radial ticks 
def draw_radial_ticks(img, center, radius, color, num_ticks=24, length=22):
    for i in range(num_ticks):
        angle = np.deg2rad(i * (360 / num_ticks))
        x1 = int(center[0] + (radius - length) * np.cos(angle))
        y1 = int(center[1] + (radius - length) * np.sin(angle))
        x2 = int(center[0] + radius * np.cos(angle))
        y2 = int(center[1] + radius * np.sin(angle))
        cv2.line(img, (x1, y1), (x2, y2), color, 3)

#drawing hud details 
def hud_details(img, center, radius, color):
    for i in range(8):
        angle = np.deg2rad(210 + i * 10)
        x1 = int(center[0] + (radius - 15) * np.cos(angle))
        y1 = int(center[1] + (radius - 15) * np.sin(angle))
        x2 = int(center[0] + (radius + 20) * np.cos(angle))
        y2 = int(center[1] + (radius + 20) * np.sin(angle))
        cv2.line(img, (x1, y1), (x2, y2), color, 4)

    for i in range(4):
        angle = np.deg2rad(270 + i * 15)
        x1 = int(center[0] + (radius - 35) * np.cos(angle))
        y1 = int(center[1] + (radius - 35) * np.sin(angle))

        cv2.rectangle(
            img,
            (x1 - 10, y1 - 10),   # FIX
            (x1 + 10, y1 + 10),   # FIX
            color,
            3
        )

#drawing more hud details (adding hud arc)
def draw_hud_arc(img, center):

    cv2.ellipse(img, tuple(center), (110, 110), 0, 330, 570, RED, 2)
    cv2.ellipse(img, tuple(center), (100, 100), 0, 330, 570, ORANGE, 3)
    cv2.ellipse(img, tuple(center), (80, 80), 0, 0, 360, CYAN, 4)

#drawing core hud 
def core_hud(img, center, radius):

    for rad_angle in np.linspace(0, 2 * np.pi, 60):
        x = radius * (0.7 + 0.3 * np.sin(6*rad_angle))
        x1 = int(center[0] + x * np.cos(rad_angle))
        y1 = int(center[1] + x * np.sin(rad_angle))
        cv2.circle(img, (x1, y1), 3, CORE, -1)

    cv2.circle(img, center, int(radius * 0.6), RED, 3)
    cv2.circle(img, center, int(radius * 0.4), CYAN, 2)


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

            #detect pinch 
            pinch_dist=np.linalg.norm(np.array(hand[4])-np.array(hand[8]))
            pinch_val=int(100-min(pinch_dist,100))

            if mean_dists > 70:
                draw_glow_circle(frame, tuple(palm), 120, CYAN, 3, glow=30)
                draw_glow_circle(frame, tuple(palm), 90, CYAN, 2, glow=20)
                draw_glow_circle(frame, tuple(palm), 60, ORANGE, 2, glow=10)
                draw_radial_ticks(frame, palm, 120, CYAN)
                hud_details(frame, palm, 155, CYAN)
                core_hud(frame, tuple(palm), 40)
                draw_hud_arc(frame, palm)
            
            # Pinch gesture: show orange arcs and value
            if pinch_val<60:
                draw_glow_circle(frame, palm, 60, ORANGE, 3, glow=20)
                cv2.putText(frame, f'Pinch: {pinch_val}', (palm[0]-40, palm[1]-70), cv2.FONT_HERSHEY_SIMPLEX, 1, ORANGE, 2)
                # for i in range(5):
                #     cv2.ellipse(frame, (palm[0]+80, palm[1]), (30,30), 0, 180, 180+pinch_val+i*10, ORANGE, 2)

            else:
                # Fist: simple glowing circle
                draw_glow_circle(frame, palm, 60, CYAN, 3, glow=20)
                cv2.putText(frame, 'FIST', (palm[0]-30, palm[1]-70), cv2.FONT_HERSHEY_SIMPLEX, 1, ORANGE, 3)
                
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
