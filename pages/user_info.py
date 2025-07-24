import streamlit as st

# ----------------------------
# Page Config
st.set_page_config(page_title="Welcome to Enhanced Eye Strain App", layout="centered")
st.title("👋 Welcome to the Enhanced Eye Strain Monitoring App")

# ----------------------------
# Intro Message
st.markdown("""
This enhanced app monitors **eye strain** and **emotions** in real time using your webcam.

### 🆕 New Features:
- **Advanced Blink Detection** with personalized thresholds
- **Facial Emotion Recognition** (7 emotions: Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral)
- **Smart Eye Strain Alerts** based on multiple indicators
- **Detailed Analytics** with session tracking

### 🔍 Enhanced Calibration Process:
1. **Normal Blinking** (15 seconds) - Establish your baseline
2. **Eye Strain Simulation** (15 seconds) - Capture tired eye patterns  
3. **Rapid Blinking** (15 seconds) - Detect irritation patterns
4. **Automatic Threshold Calculation** - Personalized detection settings

### 🎯 What We Measure:
- **Eye Aspect Ratio (EAR)** - How open your eyes are
- **Blink Rate & Duration** - Frequency and timing of blinks
- **Eye Movement Patterns** - Saccades and fixations
- **Facial Expressions** - Emotional state detection
- **Distance from Camera** - Optimal positioning feedback

### 🔒 Privacy & Data:
- All data stored locally on your device
- No cloud uploads or external sharing
- Data used only for personalization
- You can delete your data anytime

### 📋 Requirements:
- Good lighting for accurate detection
- Camera positioned at eye level
- Stable internet connection (for initial setup)
- Webcam access permissions

### ⚙️ Technical Features:
- MediaPipe facial landmark detection
- Machine learning emotion recognition
- Real-time processing (30+ FPS)
- Customizable sensitivity settings
- Session data export capabilities
""")

# ----------------------------
# System Requirements Check
st.markdown("### 🔧 System Check")

import cv2
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        st.success("✅ Camera access: Working")
        cap.release()
    else:
        st.error("❌ Camera access: Failed - Please check permissions")
except:
    st.error("❌ OpenCV not available")

# Check for emotion model
import os
if os.path.exists("face_model.h5"):
    st.success("✅ Emotion model: Found")
else:
    st.warning("⚠️ Emotion model not found - Place face_model.h5 in the app directory")

# ----------------------------
# Button to begin calibration
st.markdown("---")
if st.button("🚀 Begin Enhanced Calibration", type="primary", use_container_width=True):
    st.switch_page("pages/enhanced_calibration.py")

# Optional: Quick start for returning users
if st.button("⚡ Skip to Monitoring (use defaults)", help="Only for experienced users"):
    st.switch_page("pages/enhanced_monitor.py")