import numpy as np

# Landmark indices
THUMB_TIP = 4
THUMB_IP = 3
INDEX_TIP = 8
INDEX_MCP = 5
MIDDLE_TIP = 12
MIDDLE_MCP = 9
PINKY_MCP = 17
WRIST = 0

FINGER_TIPS = [8, 12, 16, 20]
FINGER_MCP = [5, 9, 13, 17]


def hand_scale(hand):
    return np.linalg.norm(
        np.array(hand[INDEX_MCP]) - np.array(hand[PINKY_MCP])
    )


def is_finger_extended(hand, tip, mcp):
    return hand[tip][1] < hand[mcp][1]


def are_other_fingers_folded(hand):
    """
    Index, ring, pinky folded
    """
    for tip, mcp in zip(FINGER_TIPS, FINGER_MCP):
        if hand[tip][1] < hand[mcp][1]:
            return False
    return True


# ---------- PINCH ----------
def detect_pinch(hand):
    pinch_dist = np.linalg.norm(
        np.array(hand[THUMB_TIP]) - np.array(hand[INDEX_TIP])
    )
    scale = hand_scale(hand)
    pinch_ratio = pinch_dist / scale

    pinch_strength = int(np.clip((1 - pinch_ratio) * 100, 0, 100))

    return pinch_ratio < 0.35, pinch_strength


# ---------- FIST ----------
def detect_fist(hand):
    for tip, mcp in zip(FINGER_TIPS, FINGER_MCP):
        if hand[tip][1] < hand[mcp][1]:
            return False
    return True


# ---------- OPEN HAND ----------
def detect_open(hand):
    for tip, mcp in zip(FINGER_TIPS, FINGER_MCP):
        if hand[tip][1] > hand[mcp][1]:
            return False
    return True


# ---------- THUMBS UP / DOWN ----------
def detect_thumb_gesture(hand):
    thumb_tip = hand[THUMB_TIP]
    thumb_ip = hand[THUMB_IP]
    wrist = hand[WRIST]

    thumb_extended = thumb_tip[1] < thumb_ip[1]
    others_folded = are_other_fingers_folded(hand)

    if not (thumb_extended and others_folded):
        return None

    if thumb_tip[1] < wrist[1] - 20:
        return "THUMBS_UP"

    if thumb_tip[1] > wrist[1] + 20:
        return "THUMBS_DOWN"

    return None


# ---------- MIDDLE FINGER (INAPPROPRIATE) ----------
def detect_middle_finger(hand):
    """
    Middle finger extended, others folded
    """
    middle_extended = is_finger_extended(hand, MIDDLE_TIP, MIDDLE_MCP)

    index_folded = hand[8][1] > hand[5][1]
    ring_folded = hand[16][1] > hand[13][1]
    pinky_folded = hand[20][1] > hand[17][1]

    if middle_extended and index_folded and ring_folded and pinky_folded:
        return True

    return False


# ---------- CLASSIFIER ----------
def classify_gesture(hand):
    pinch, strength = detect_pinch(hand)
    if pinch:
        return "PINCH", strength

    thumb = detect_thumb_gesture(hand)
    if thumb:
        return thumb, None

    if detect_middle_finger(hand):
        return "MIDDLE_FINGER", None

    if detect_fist(hand):
        return "FIST", None

    if detect_open(hand):
        return "OPEN", None

    return "UNKNOWN", None
