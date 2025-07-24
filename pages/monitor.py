import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(page_title="Enhanced Live Eye Monitor with Face Expression Recognition", layout="wide")

import cv2
import numpy as np
import time
import mediapipe as mp
from plyer import notification
import pandas as pd
import os
import json
from datetime import datetime

# Try to import DeepFace for emotion detection
EMOTION_AVAILABLE = False

try:
    from deepface import DeepFace

    EMOTION_AVAILABLE = True
    st.success("✅ Face library loaded successfully for emotion detection")
except ImportError:
    st.error("❌ Face library not found!")
except Exception as e:
    st.warning(f"⚠️ Face library error: {str(e)}")

st.title("🔴 Enhanced Eye Strain & Emotion Monitoring")

# Get user info
user_id = st.session_state.get("user_id")
user_folder = st.session_state.get("user_folder")
if not user_id or not user_folder:
    st.error("User not logged in. Please go back to home.")
    st.stop()

# ===========================
# CONFIGURATION
# ===========================

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indices (MediaPipe)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Load user thresholds
thresholds_file = os.path.join(user_folder, "thresholds.json")
if os.path.exists(thresholds_file):
    with open(thresholds_file, 'r') as f:
        user_thresholds = json.load(f)
    st.success(f"✅ Using personalized thresholds from calibration")
else:
    user_thresholds = {
        'blink_threshold': 0.25,
        'strain_threshold': 0.22,
        'normal_ear_mean': 0.3
    }
    st.info("ℹ️ Using default thresholds. Consider running calibration for better accuracy.")


# ===========================
# HELPER FUNCTIONS
# ===========================

def calculate_ear(landmarks, eye_indices, width, height):
    """Calculate Eye Aspect Ratio (EAR) for blink detection"""
    try:
        coords = [(int(landmarks[i].x * width), int(landmarks[i].y * height)) for i in eye_indices]

        # Vertical distances
        A = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
        B = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))

        # Horizontal distance
        C = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))

        # Calculate EAR
        ear = (A + B) / (2.0 * C)
        return max(0.0, min(1.0, ear))  # Clamp between 0 and 1
    except:
        return 0.0


def detect_emotion_deepface(frame):
    """Detect emotion using DeepFace library"""
    if not EMOTION_AVAILABLE:
        return "Face Not Available", 0.0

    try:
        # Convert RGB to BGR for DeepFace (OpenCV format)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Analyze emotions with DeepFace
        result = DeepFace.analyze(
            frame_bgr,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv',  # Use opencv for speed
            silent=True  # Suppress console output
        )

        # Handle both single face and multiple faces
        if isinstance(result, list):
            result = result[0]

        # Get emotions and dominant emotion
        emotions = result['emotion']
        dominant_emotion = result['dominant_emotion']

        # Get confidence score (percentage converted to 0-1 range)
        confidence = emotions[dominant_emotion] / 100.0

        return dominant_emotion.capitalize(), confidence

    except Exception as e:
        # Return a shortened error message
        error_msg = str(e)[:30]
        return f"Error: {error_msg}", 0.0


# Advanced blink detector class
class EnhancedBlinkDetector:
    def __init__(self, ear_threshold=0.25, consecutive_frames=3):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.ear_history = []
        self.closed_frames = 0
        self.blink_count = 0
        self.last_blink_time = 0
        self.strain_counter = 0

    def update(self, ear):
        current_time = time.time()
        self.ear_history.append(ear)

        # Keep only last 30 EAR values for smoothing
        if len(self.ear_history) > 30:
            self.ear_history.pop(0)

        # Smooth EAR using moving average
        smoothed_ear = np.mean(self.ear_history[-5:]) if len(self.ear_history) >= 5 else ear

        # Blink detection
        blink_detected = False
        if smoothed_ear < self.ear_threshold:
            self.closed_frames += 1
            # Increment strain counter for prolonged low EAR
            if self.closed_frames > 10:  # Eyes closed for more than ~333ms
                self.strain_counter += 1
        else:
            if self.closed_frames >= self.consecutive_frames:
                # Debouncing: minimum 300ms between blinks
                if current_time - self.last_blink_time > 0.3:
                    self.blink_count += 1
                    self.last_blink_time = current_time
                    blink_detected = True
            self.closed_frames = 0
            # Decrease strain counter when eyes are open normally
            self.strain_counter = max(0, self.strain_counter - 1)

        return blink_detected, smoothed_ear

    def get_stats(self):
        return {
            'blink_count': self.blink_count,
            'strain_level': self.strain_counter,
            'avg_ear': np.mean(self.ear_history) if self.ear_history else 0.0,
            'current_ear': self.ear_history[-1] if self.ear_history else 0.0
        }


# ===========================
# STREAMLIT UI
# ===========================

# Session state initialization
if "monitoring_active" not in st.session_state:
    st.session_state.monitoring_active = False
if "session_data" not in st.session_state:
    st.session_state.session_data = []
if "last_emotion_update" not in st.session_state:
    st.session_state.last_emotion_update = 0

# UI Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Camera Feed")
    video_placeholder = st.empty()

with col2:
    st.subheader("📊 Real-time Metrics")
    blink_metric = st.empty()
    emotion_metric = st.empty()
    ear_metric = st.empty()
    strain_metric = st.empty()
    session_metric = st.empty()

    st.subheader("⚙️ Settings")
    # Use user's calibrated thresholds as defaults
    ear_threshold = st.slider("EAR Threshold", 0.15, 0.35,
                              user_thresholds.get('blink_threshold', 0.25), 0.01)
    blink_frames = st.slider("Consecutive frames for blink", 2, 10, 3)
    strain_threshold = st.slider("Strain detection sensitivity", 10, 100, 50)
    alert_cooldown = st.slider("Alert cooldown (seconds)", 10, 120, 30)
    emotion_interval = st.slider("Emotion detection interval (seconds)", 2, 10, 3)

# Control buttons
button_col1, button_col2, button_col3 = st.columns(3)

with button_col1:
    start_monitoring = st.button("▶️ Start Monitoring", type="primary")

with button_col2:
    stop_monitoring = st.button("⏹️ Stop Monitoring")

with button_col3:
    if st.button("📊 Dashboard"):
        st.switch_page("pages/dashboard.py")

# Display DeepFace status
if EMOTION_AVAILABLE:
    st.success("✅ Face Model is ready for emotion detection!")
    with st.expander("🤖 Face Info"):
        st.info("""
        **Face Model Features:**
        - Advanced emotion detection using deep learning
        - 7 emotion classes: Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
        - High accuracy with confidence scores
        - Multiple face detection backends
        - Real-time processing optimized
        """)
else:
    st.error("❌ Face library not available")
    with st.expander("🔧 Installation Help"):
        st.code("pip install face library")
        st.warning("Note: First run might take longer as models are downloaded.")

# ===========================
# MAIN MONITORING LOOP
# ===========================

if start_monitoring:
    st.session_state.monitoring_active = True
    st.session_state.session_data = []  # Reset session data
    st.session_state.last_emotion_update = 0

if stop_monitoring:
    st.session_state.monitoring_active = False

if st.session_state.monitoring_active:
    # Initialize camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Could not access camera. Please check permissions.")
        st.session_state.monitoring_active = False
    else:
        # Initialize monitoring components
        blink_detector = EnhancedBlinkDetector(ear_threshold=ear_threshold,
                                               consecutive_frames=blink_frames)

        # Monitoring variables
        strain_alerts = 0
        last_alert_time = 0
        session_start = time.time()
        frame_count = 0
        current_emotion = "Initializing..."
        emotion_confidence = 0.0
        emotion_status = st.empty()

        st.success("✅ Monitoring started! Press 'Stop Monitoring' to end.")

        # Download models on first run (if needed)
        if EMOTION_AVAILABLE and frame_count == 0:
            with st.spinner("🔄 Loading Face models (first run only)..."):
                try:
                    # Test DeepFace with a dummy image to load models
                    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
                    DeepFace.analyze(dummy_img, actions=['emotion'], enforce_detection=False, silent=True)
                    st.success("✅ Face models loaded successfully!")
                except:
                    st.warning("⚠️ Face model loading issue - continuing anyway")

        # Main monitoring loop
        while st.session_state.monitoring_active:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to read from camera")
                break

            frame_count += 1
            height, width = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe for eye tracking
            results = face_mesh.process(frame_rgb)
            frame_display = frame_rgb.copy()

            # Initialize metrics
            current_ear = 0.0
            blink_detected = False

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # Calculate EAR
                left_ear = calculate_ear(landmarks, LEFT_EYE, width, height)
                right_ear = calculate_ear(landmarks, RIGHT_EYE, width, height)
                current_ear = (left_ear + right_ear) / 2.0

                # Update blink detector
                blink_detected, smoothed_ear = blink_detector.update(current_ear)
                stats = blink_detector.get_stats()

                # Strain detection based on multiple factors
                strain_level = stats['strain_level']
                if strain_level >= strain_threshold:
                    current_time = time.time()
                    if current_time - last_alert_time > alert_cooldown:
                        strain_alerts += 1
                        last_alert_time = current_time

                        # Send notification
                        try:
                            notification.notify(
                                title="Eye Strain Alert",
                                message="Take a break! Eye strain detected.",
                                timeout=5
                            )
                        except:
                            pass  # Notification might fail in some environments

                # Draw eye landmarks
                for idx in LEFT_EYE + RIGHT_EYE:
                    x = int(landmarks[idx].x * width)
                    y = int(landmarks[idx].y * height)
                    color = (255, 0, 0) if blink_detected else (0, 255, 0)
                    cv2.circle(frame_display, (x, y), 2, color, -1)

            # Emotion detection (every N seconds)
            current_time = time.time()
            if (current_time - st.session_state.last_emotion_update) >= emotion_interval and EMOTION_AVAILABLE:
                emotion_status.info("🔍 Analyzing emotion...")
                current_emotion, emotion_confidence = detect_emotion_deepface(frame_rgb)
                st.session_state.last_emotion_update = current_time
                emotion_status.empty()

            # Add text overlays to frame
            cv2.putText(frame_display, f"EAR: {current_ear:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_display, f"Blinks: {blink_detector.blink_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame_display, f"Emotion: {current_emotion}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame_display, f"Conf: {emotion_confidence:.2f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Example: Assuming emotion_pred is a string like "Happy", "Sad", etc.
            # And strain_level is computed from EAR/blink
            # You must have emotion_pred = predict_emotion(face_crop) somewhere earlier

            # Define which emotions contribute to fatigue
            fatigue_emotions = ["Sad", "Angry", "Fear", "Disgust"]

            # Combine logic
            if strain_level >= strain_threshold and emotion_pred in fatigue_emotions:
                status_text = "STRAIN DETECTED!"
                status_color = (0, 0, 255)  # Red
            elif strain_level >= strain_threshold / 2 and emotion_pred in fatigue_emotions:
                status_text = "Elevated Strain"
                status_color = (0, 255, 255)  # Yellow
            else:
                status_text = "Normal"
                status_color = (0, 255, 0)  # Green

            cv2.putText(frame_display, f"Status: {status_text}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)


            # Blink indicator
            if blink_detected:
                cv2.putText(frame_display, "BLINK!", (width // 2 - 50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            # DeepFace status indicator
            deepface_color = (0, 255, 0) if EMOTION_AVAILABLE else (255, 0, 0)
            deepface_text = "Face Model: ON" if EMOTION_AVAILABLE else "Face Model: OFF"
            cv2.putText(frame_display, deepface_text, (width - 180, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, deepface_color, 2)

            # Update display
            video_placeholder.image(frame_display, channels="RGB")

            # Update metrics
            blink_metric.metric("👁️ Blinks Detected", blink_detector.blink_count)
            emotion_metric.metric("😊 Current Emotion", f"{current_emotion} ({emotion_confidence:.2f})")
            ear_metric.metric("👀 Eye Aspect Ratio", f"{current_ear:.3f}")
            strain_metric.metric("⚠️ Strain Level", f"{strain_level}/{strain_threshold}")

            # Session duration
            session_duration = (time.time() - session_start) / 60
            session_metric.metric("⏱️ Session Duration", f"{session_duration:.1f} min")

            # Log data every 60 frames (~2 seconds at 30 FPS)
            if frame_count % 60 == 0:
                session_data = {
                    'timestamp': datetime.now(),
                    'ear': current_ear,
                    'blink_count': blink_detector.blink_count,
                    'emotion': current_emotion,
                    'emotion_confidence': emotion_confidence,
                    'strain_level': strain_level,
                    'strain_alerts': strain_alerts,
                    'session_duration_min': session_duration
                }
                st.session_state.session_data.append(session_data)

            time.sleep(0.033)  # ~30 FPS

        cap.release()
        cv2.destroyAllWindows()
        emotion_status.empty()

        # Save session data
        if st.session_state.session_data:
            session_df = pd.DataFrame(st.session_state.session_data)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            session_file = os.path.join(user_folder, f"session_{timestamp}.csv")
            session_df.to_csv(session_file, index=False)
            st.success(f"📁 Session data saved: {len(st.session_state.session_data)} data points")

# ===========================
# SESSION SUMMARY
# ===========================

if not st.session_state.monitoring_active and st.session_state.session_data:
    st.markdown("---")
    st.subheader("📈 Session Summary")

    df = pd.DataFrame(st.session_state.session_data)

    # Summary metrics
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

    with summary_col1:
        avg_ear = df['ear'].mean()
        st.metric("Average EAR", f"{avg_ear:.3f}")

    with summary_col2:
        final_blinks = df['blink_count'].iloc[-1] if len(df) > 0 else 0
        duration_minutes = df['session_duration_min'].iloc[-1] if len(df) > 0 else 0
        blink_rate = final_blinks / duration_minutes if duration_minutes > 0 else 0
        st.metric("Blink Rate", f"{blink_rate:.1f}/min")

    with summary_col3:
        if 'emotion' in df.columns and len(df) > 0:
            # Filter out initialization and error messages
            valid_emotions = ['Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Disgust', 'Neutral']
            emotion_data = df[df['emotion'].isin(valid_emotions)]
            if not emotion_data.empty:
                most_common_emotion = emotion_data['emotion'].mode()[0]
                emotion_changes = (emotion_data['emotion'] != emotion_data['emotion'].shift()).sum()
                st.metric("Dominant Emotion", f"{most_common_emotion} ({emotion_changes} changes)")
            else:
                st.metric("Dominant Emotion", "Insufficient data")

    with summary_col4:
        max_strain_level = df['strain_level'].max()
        total_alerts = df['strain_alerts'].iloc[-1] if len(df) > 0 else 0
        st.metric("Max Strain / Alerts", f"{max_strain_level} / {total_alerts}")

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        # EAR over time
        st.line_chart(df.set_index('timestamp')['ear'], height=250)
        st.caption("Eye Aspect Ratio over time")

    with col2:
        # Strain level over time
        st.line_chart(df.set_index('timestamp')['strain_level'], height=250)
        st.caption("Strain level over time")

    # Emotion analysis
    if 'emotion' in df.columns:
        valid_emotions = ['Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Disgust', 'Neutral']
        emotion_data = df[df['emotion'].isin(valid_emotions)]

        if not emotion_data.empty:
            col1, col2 = st.columns(2)

            with col1:
                # Emotion distribution
                emotion_counts = emotion_data['emotion'].value_counts()
                st.bar_chart(emotion_counts, height=200)
                st.caption("Emotion distribution during session")

            with col2:
                # Emotion confidence over time
                st.line_chart(emotion_data.set_index('timestamp')['emotion_confidence'], height=200)
                st.caption("Emotion confidence over time")

            # Detailed emotion statistics
            st.subheader("📊 Detailed Emotion Analysis")
            emotion_stats = emotion_data.groupby('emotion').agg({
                'emotion_confidence': ['mean', 'min', 'max', 'count']
            }).round(3)
            emotion_stats.columns = ['Avg Confidence', 'Min Confidence', 'Max Confidence', 'Detection Count']
            st.dataframe(emotion_stats)

# ===========================
# TROUBLESHOOTING & TIPS
# ===========================

with st.expander("🔧 Face Model Troubleshooting & Tips"):
    st.markdown("""
    ### Optimizing Performance:
    - **Lighting**: Ensure good, even lighting on your face
    - **Distance**: Sit 1-2 feet from the camera for best results
    - **Angle**: Face the camera directly for accurate detection
    - **Interval**: Increase emotion detection interval for better performance

    ### Understanding Emotions:
    - **Happy**: Smiling, positive expressions
    - **Sad**: Downward mouth, droopy eyes
    - **Angry**: Furrowed brow, tense expression
    - **Fear**: Wide eyes, surprised/concerned look
    - **Surprise**: Raised eyebrows, open mouth
    - **Disgust**: Wrinkled nose, negative expression
    - **Neutral**: Relaxed, no strong expression

    ### Confidence Scores:
    - Values from 0.0 to 1.0 (higher = more confident)
    - Scores above 0.7 are generally reliable
    - Multiple expressions may show lower confidence

    ### Common Issues:
    - **"No face detected"**: Improve lighting or move closer
    - **Low confidence scores**: Make clearer expressions
    - **Inconsistent results**: Ensure stable head position
    - **Slow processing**: Increase emotion detection interval
    """)

# ===========================
# SYSTEM DIAGNOSTICS
# ===========================

with st.expander("🔍 System Diagnostics"):
    st.markdown("### System Status")

    diag_col1, diag_col2 = st.columns(2)

    with diag_col1:
        st.write("**Hardware Status:**")

        # Camera test
        cap_test = cv2.VideoCapture(0)
        if cap_test.isOpened():
            st.success("✅ Camera: Available")
            ret, frame = cap_test.read()
            if ret:
                h, w = frame.shape[:2]
                st.write(f"Resolution: {w}x{h}")
            cap_test.release()
        else:
            st.error("❌ Camera: Not available")

        # MediaPipe status
        st.success("✅ MediaPipe: Ready")

    with diag_col2:
        st.write("**Software Status:**")

        # DeepFace status
        if EMOTION_AVAILABLE:
            st.success("✅ Face Model: Available")

            # Test DeepFace
            if st.button("🧪 Test Face Model"):
                with st.spinner("Testing Face Model..."):
                    try:
                        test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                        result = DeepFace.analyze(test_img, actions=['emotion'], enforce_detection=False, silent=True)
                        st.success("✅ Face Model test successful")
                    except Exception as e:
                        st.error(f"❌ Face Model test failed: {str(e)}")
        else:
            st.error("❌ Face Model: Not available")

        # Calibration status
        if os.path.exists(thresholds_file):
            st.success("✅ Calibration: Complete")
        else:
            st.warning("⚠️ Calibration: Recommended")

    # Performance metrics
    if st.session_state.session_data:
        st.markdown("### Performance Metrics")
        df_perf = pd.DataFrame(st.session_state.session_data)

        perf_col1, perf_col2, perf_col3 = st.columns(3)

        with perf_col1:
            st.metric("Total Frames Processed", len(df_perf))

        with perf_col2:
            valid_emotions = len(
                df_perf[df_perf['emotion'].isin(['Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Disgust', 'Neutral'])])
            st.metric("Successful Emotion Detections", valid_emotions)

        with perf_col3:
            avg_confidence = df_perf['emotion_confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.2f}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<small>Enhanced Eye Strain Monitor with FaceModel | Advanced emotion detection<br>
Powered by VRAY - State-of-the-art facial analysis | Take care of your eyes! 👁️</small>
</div>
""", unsafe_allow_html=True)