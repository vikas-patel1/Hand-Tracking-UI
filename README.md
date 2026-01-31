# 🖐️ Hand Tracking UI – Real-Time Gesture-Driven Interface

Hand Tracking UI is a **real-time computer vision project** built using **Python, OpenCV, and MediaPipe** that detects human hands from a webcam feed, extracts precise hand landmarks, and renders **dynamic futuristic UI overlays** that move smoothly with the hand.

This project is designed as a **high-quality portfolio-level computer vision application**, demonstrating practical skills in real-time video processing, modular Python architecture, and interactive UI rendering — without requiring machine learning training or GPUs.

---

## 📌 Project Description

The system captures live video frames from a webcam, processes each frame to detect hands, computes **21 anatomical landmarks per hand**, and overlays custom HUD/UI elements anchored to these landmarks.

The focus of this project is:
- Real-time performance
- Clean separation of concerns (tracking, UI, orchestration)
- Visual clarity and responsiveness
- Extendability for gesture-based applications

The project runs entirely on CPU using a **pre-trained hand landmark model**, making it lightweight and easy to run on most systems.

---

## 🎯 Project Goals

- Track hands accurately in real time
- Extract detailed hand landmark coordinates
- Render UI overlays that follow hand motion
- Maintain smooth frame rates and low latency
- Provide a solid base for gesture-controlled systems

---

## ✨ Features

- 🖐 Real-time hand detection via webcam  
- 📍 21 landmark points per detected hand  
- 🎨 Futuristic UI / HUD overlay rendering  
- ⚡ Smooth and responsive real-time processing  
- 🧩 Modular and extensible Python codebase  
- 💻 Runs entirely on CPU (no GPU required)

---

## 🗂️ Project Structure

Hand-Tracking-UI/
│
├── main.py # Main application controller
├── hand_tracker.py # Hand detection & landmark extraction
├── hud.py # UI / HUD rendering logic
├── requirements.txt # Python dependencies
└── README.md # Complete project documentation


---

## 🧠 How the System Works

### 1️⃣ Video Capture
- Webcam feed is accessed using OpenCV
- Frames are captured continuously in a loop
- Frames are prepared for processing in real time

### 2️⃣ Hand Detection & Landmark Extraction
- MediaPipe’s hand model detects hands in each frame
- For every detected hand, **21 landmarks** are extracted
- Each landmark contains `(x, y, z)` coordinates
- These points represent fingers, joints, and wrist

### 3️⃣ UI / HUD Rendering
- Landmark data is passed to the HUD module
- UI elements are drawn relative to landmark positions
- Overlays dynamically follow hand movement
- Visuals remain stable and smooth

### 4️⃣ Display Output
- Final composed frame is displayed in real time
- The system maintains consistent FPS for smooth interaction

---

## ⚙️ Technologies Used

| Technology | Role |
|----------|------|
| Python | Core programming language |
| OpenCV | Video capture, image processing, rendering |
| MediaPipe | Hand detection & landmark estimation |
| NumPy | Mathematical computations |

---

## 🖥️ System Requirements

- Python **3.8 or higher**
- Webcam (built-in or USB)
- Windows / macOS / Linux
- Minimum **4 GB RAM** recommended

---

## 📦 Installation Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/vikas-patel1/Hand-Tracking-UI.git
cd Hand-Tracking-UI

2️⃣ Create Virtual Environment (Recommended)
-python -m venv venv


Activate it:

Windows
-venv\Scripts\activate

Linux / macOS
-source venv/bin/activate

3️⃣ Install Dependencies
-pip install -r requirements.txt

▶️ Running the Application
-python main.py

What Happens Next:

-Webcam window opens

-Place your hand in front of the camera

-Hand landmarks and UI overlays appear instantly

-Press Q to exit

🛠 Customization & Extension Ideas

This project can be extended to:
-✋ Gesture recognition (pinch, swipe, click)
-🖥 Gesture-controlled UI systems
-🎮 Gesture-based games
-🥽 AR / VR interaction layers
-🔊 System or application controls