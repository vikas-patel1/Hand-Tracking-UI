import cv2
import numpy as np

# Colors
CYAN = (255, 255, 0)
ORANGE = (0, 180, 255)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
CORE = (0, 255, 180)


def draw_glow_circle(frame, center, radius, color, thickness=2, glow=15):
    for g in range(glow, 0, -3):
        alpha = 0.08 + 0.12 * (g / glow)
        overlay = frame.copy()
        cv2.circle(overlay, center, radius + g, color, thickness)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.circle(frame, center, radius, color, thickness)


def draw_radial_ticks(img, center, radius, color, num_ticks=24, length=22):
    for i in range(num_ticks):
        angle = np.deg2rad(i * (360 / num_ticks))
        x1 = int(center[0] + (radius - length) * np.cos(angle))
        y1 = int(center[1] + (radius - length) * np.sin(angle))
        x2 = int(center[0] + radius * np.cos(angle))
        y2 = int(center[1] + radius * np.sin(angle))
        cv2.line(img, (x1, y1), (x2, y2), color, 3)


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
        cv2.rectangle(img, (x1 - 10, y1 - 10), (x1 + 10, y1 + 10), color, 3)


def draw_hud_arc(img, center):
    cv2.ellipse(img, tuple(center), (110, 110), 0, 330, 570, RED, 2)
    cv2.ellipse(img, tuple(center), (100, 100), 0, 330, 570, ORANGE, 3)
    cv2.ellipse(img, tuple(center), (80, 80), 0, 0, 360, CYAN, 4)


def core_hud(img, center, radius):
    for rad_angle in np.linspace(0, 2 * np.pi, 60):
        x = radius * (0.7 + 0.3 * np.sin(6 * rad_angle))
        x1 = int(center[0] + x * np.cos(rad_angle))
        y1 = int(center[1] + x * np.sin(rad_angle))
        cv2.circle(img, (x1, y1), 3, CORE, -1)

    cv2.circle(img, center, int(radius * 0.6), RED, 3)
    cv2.circle(img, center, int(radius * 0.4), CYAN, 2)
