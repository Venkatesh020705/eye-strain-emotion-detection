import streamlit as st
import cv2
import time
import numpy as np
import pandas as pd
import os
import mediapipe as mp
from datetime import datetime
import json

# --- Page Config ---
st.set_page_config(page_title="Enhanced Eye Calibration", layout="wide")
st.title("👁️ Enhanced Eye Calibration: Blink Detection & Data Collection")

user_id = st.session_state.get("user_id")
user_folder = st.session_state.get("user_folder")
if not user_id or not user_folder:
    st.error("User not logged in. Please return to home.")
    st.stop()

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    video_placeholder = st.empty()
with col2:
    info_placeholder = st.empty()
    metrics_placeholder = st.empty()

# --- MediaPipe Setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# --- Landmark Indices ---
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# --- Improved EAR Calculation ---
def calculate_ear(landmarks, eye_indices, w, h):
    """Calculate Eye Aspect Ratio with improved accuracy"""
    try:
        coords = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
        points = np.array(coords)
        
        # Calculate vertical distances
        vertical_1 = np.linalg.norm(points[1] - points[5])
        vertical_2 = np.linalg.norm(points[2] - points[4])
        
        # Calculate horizontal distance
        horizontal = np.linalg.norm(points[0] - points[3])
        
        # EAR formula
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return max(0.0, min(1.0, ear))  # Clamp between 0 and 1
    except Exception as e:
        return 0.0

# --- Advanced Blink Detection ---
class BlinkDetector:
    def __init__(self, ear_threshold=0.25, consecutive_frames=3):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.ear_history = []
        self.closed_frames = 0
        self.blink_count = 0
        self.last_blink_time = 0
        
    def update(self, ear):
        """Update blink detector with new EAR value"""
        current_time = time.time()
        self.ear_history.append(ear)
        
        # Keep only last 30 EAR values for smoothing
        if len(self.ear_history) > 30:
            self.ear_history.pop(0)
        
        # Smooth EAR using moving average
        smoothed_ear = np.mean(self.ear_history[-5:]) if len(self.ear_history) >= 5 else ear
        
        # Blink detection logic
        if smoothed_ear < self.ear_threshold:
            self.closed_frames += 1
        else:
            # Check if we had enough consecutive closed frames for a blink
            if self.closed_frames >= self.consecutive_frames:
                # Avoid counting rapid consecutive blinks (debouncing)
                if current_time - self.last_blink_time > 0.3:  # 300ms minimum between blinks
                    self.blink_count += 1
                    self.last_blink_time = current_time
                    return True  # Blink detected
            self.closed_frames = 0
        
        return False  # No blink
    
    def get_stats(self):
        """Get blink statistics"""
        return {
            'blink_count': self.blink_count,
            'avg_ear': np.mean(self.ear_history) if self.ear_history else 0.0,
            'current_ear': self.ear_history[-1] if self.ear_history else 0.0
        }

# --- Calibration Function ---
def auto_calibrate_ear(cap, duration=5):
    """Automatically calibrate EAR threshold"""
    info_placeholder.info("🔧 Auto-calibrating... Keep your eyes open normally.")
    
    ear_values = []
    start_time = time.time()
    
    progress_bar = st.progress(0)
    
    while time.time() - start_time < duration:
        progress = (time.time() - start_time) / duration
        progress_bar.progress(progress)
        
        ret, frame = cap.read()
        if not ret:
            continue
        
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
            right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Only collect valid EAR values
            if 0.15 < avg_ear < 0.45:
                ear_values.append(avg_ear)
        
        # Show preview during calibration
        frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if results.multi_face_landmarks:
            # Draw eye landmarks
            landmarks = results.multi_face_landmarks[0].landmark
            for idx in LEFT_EYE + RIGHT_EYE:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame_display, (x, y), 2, (0, 255, 0), -1)
        
        video_placeholder.image(frame_display, channels="RGB")
        time.sleep(0.033)  # ~30 FPS
    
    progress_bar.empty()
    
    if ear_values:
        mean_ear = np.mean(ear_values)
        std_ear = np.std(ear_values)
        # Set threshold at mean - 2*std (captures most blinks)
        threshold = max(0.15, mean_ear - 2*std_ear)
        info_placeholder.success(f"✅ Calibration complete! Mean EAR: {mean_ear:.3f}, Threshold: {threshold:.3f}")
        return threshold
    else:
        info_placeholder.warning("⚠️ Could not collect enough valid EAR values. Using default.")
        return 0.25

# --- Main Data Collection Function ---
def collect_enhanced_data(label, phase_text, duration=15):
    """Enhanced data collection with better blink detection"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Could not access camera")
        return []
    
    # Auto-calibrate EAR threshold
    ear_threshold = auto_calibrate_ear(cap, duration=5)
    
    # Initialize detector with calibrated threshold
    blink_detector = BlinkDetector(ear_threshold=ear_threshold, consecutive_frames=3)
    
    info_placeholder.info(f"{phase_text}: Position yourself comfortably and begin...")
    
    # Data collection variables
    features = []
    start_time = time.time()
    
    # Eye movement tracking
    prev_eye_positions = []
    saccade_count = 0
    fixation_start = None
    fixation_durations = []
    
    progress_bar = st.progress(0)
    
    while time.time() - start_time < duration:
        elapsed = time.time() - start_time
        progress = elapsed / duration
        progress_bar.progress(progress)
        
        ret, frame = cap.read()
        if not ret:
            continue
        
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        frame_display = rgb.copy()
        
        # Initialize frame data
        frame_data = {
            'timestamp': round(elapsed, 2),
            'ear_left': 0.0,
            'ear_right': 0.0,
            'ear_avg': 0.0,
            'blink_detected': False,
            'total_blinks': 0,
            'eye_distance': 0.0,
            'saccade_count': 0,
            'fixation_duration': 0.0,
            'face_detected': False,
            'label': label
        }
        
        if results.multi_face_landmarks:
            frame_data['face_detected'] = True
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Calculate EAR for both eyes
            left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
            right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            
            frame_data['ear_left'] = left_ear
            frame_data['ear_right'] = right_ear
            frame_data['ear_avg'] = avg_ear
            
            # Update blink detector
            blink_detected = blink_detector.update(avg_ear)
            frame_data['blink_detected'] = blink_detected
            frame_data['total_blinks'] = blink_detector.blink_count
            
            # Calculate eye distance (for distance from camera)
            left_corner = np.array([landmarks[33].x, landmarks[33].y])
            right_corner = np.array([landmarks[263].x, landmarks[263].y])
            eye_distance = np.linalg.norm(left_corner - right_corner) * w
            frame_data['eye_distance'] = eye_distance
            
            # Track eye movements for saccades/fixations
            eye_center = (left_corner + right_corner) / 2
            if prev_eye_positions:
                movement = np.linalg.norm(eye_center - prev_eye_positions[-1])
                if movement > 0.02:  # Threshold for saccade
                    saccade_count += 1
                    if fixation_start:
                        fixation_durations.append(time.time() - fixation_start)
                        fixation_start = None
                else:
                    if not fixation_start:
                        fixation_start = time.time()
            
            prev_eye_positions.append(eye_center)
            if len(prev_eye_positions) > 10:  # Keep only recent positions
                prev_eye_positions.pop(0)
            
            frame_data['saccade_count'] = saccade_count
            frame_data['fixation_duration'] = np.mean(fixation_durations) if fixation_durations else 0.0
            
            # Draw eye landmarks on frame
            for idx in LEFT_EYE + RIGHT_EYE:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                color = (255, 0, 0) if blink_detected else (0, 255, 0)
                cv2.circle(frame_display, (x, y), 2, color, -1)
            
            # Draw blink indicator
            if blink_detected:
                cv2.putText(frame_display, "BLINK!", (w//2-50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Add text overlays
        cv2.putText(frame_display, f"{phase_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_display, f"Time: {int(duration - elapsed)}s", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame_display, f"Blinks: {frame_data['total_blinks']}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame_display, f"EAR: {frame_data['ear_avg']:.3f}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Update display
        video_placeholder.image(frame_display, channels="RGB")
        
        # Update metrics in sidebar
        stats = blink_detector.get_stats()
        with metrics_placeholder.container():
            st.metric("Blinks Detected", stats['blink_count'])
            st.metric("Current EAR", f"{stats['current_ear']:.3f}")
            st.metric("Average EAR", f"{stats['avg_ear']:.3f}")
        
        # Store frame data
        features.append(frame_data)
        
        # Small delay for better performance
        time.sleep(0.033)  # ~30 FPS
    
    progress_bar.empty()
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    total_blinks = blink_detector.blink_count
    blink_rate = total_blinks / (duration / 60)  # blinks per minute
    
    info_placeholder.success(f"✅ {phase_text} Complete! Total blinks: {total_blinks}, Rate: {blink_rate:.1f}/min")
    
    return features

# --- Streamlit Interface ---
st.markdown("### 📋 Calibration Steps")

# Initialize session state
if "calibration_step" not in st.session_state:
    st.session_state.calibration_step = 1

if "phase1_data" not in st.session_state:
    st.session_state.phase1_data = []

if "phase2_data" not in st.session_state:
    st.session_state.phase2_data = []

if "phase3_data" not in st.session_state:
    st.session_state.phase3_data = []

# Step indicators
step_cols = st.columns(4)
with step_cols[0]:
    status = "✅" if st.session_state.calibration_step > 1 else "🔘"
    st.write(f"{status} Step 1: Normal Blinking")

with step_cols[1]:
    status = "✅" if st.session_state.calibration_step > 2 else "🔘"
    st.write(f"{status} Step 2: Tired/Strain Simulation")

with step_cols[2]:
    status = "✅" if st.session_state.calibration_step > 3 else "🔘"
    st.write(f"{status} Step 3: Rapid Blinking")

with step_cols[3]:
    status = "✅" if st.session_state.calibration_step > 4 else "🔘"
    st.write(f"{status} Step 4: Save & Analysis")

st.markdown("---")

# Step 1: Normal Blinking
if st.session_state.calibration_step == 1:
    st.header("Step 1: Normal Blinking Pattern")
    st.info("📝 Instructions: Sit comfortably and blink naturally. Try to maintain normal eye behavior as if you were reading or working.")
    
    col1, col2 = st.columns(2)
    with col1:
        duration = st.slider("Duration (seconds)", 10, 30, 15)
    with col2:
        if st.button("▶️ Start Normal Blinking Collection", type="primary"):
            st.session_state.phase1_data = collect_enhanced_data(
                label=0, 
                phase_text="Normal Blinking", 
                duration=duration
            )
            if st.session_state.phase1_data:
                st.session_state.calibration_step = 2
                st.rerun()

# Step 2: Tired/Strain Simulation
elif st.session_state.calibration_step == 2:
    st.header("Step 2: Eye Strain Simulation")
    st.info("📝 Instructions: Simulate eye fatigue by keeping your eyes partially closed, blinking slowly, or squinting slightly as if you're tired.")
    
    col1, col2 = st.columns(2)
    with col1:
        duration = st.slider("Duration (seconds)", 10, 30, 15)
    with col2:
        if st.button("▶️ Start Strain Simulation", type="primary"):
            st.session_state.phase2_data = collect_enhanced_data(
                label=1, 
                phase_text="Eye Strain Simulation", 
                duration=duration
            )
            if st.session_state.phase2_data:
                st.session_state.calibration_step = 3
                st.rerun()

# Step 3: Rapid Blinking
elif st.session_state.calibration_step == 3:
    st.header("Step 3: Rapid Blinking Pattern")
    st.info("📝 Instructions: Blink more frequently than normal, as might happen when eyes are dry or irritated.")
    
    col1, col2 = st.columns(2)
    with col1:
        duration = st.slider("Duration (seconds)", 10, 30, 15)
    with col2:
        if st.button("▶️ Start Rapid Blinking Collection", type="primary"):
            st.session_state.phase3_data = collect_enhanced_data(
                label=2, 
                phase_text="Rapid Blinking", 
                duration=duration
            )
            if st.session_state.phase3_data:
                st.session_state.calibration_step = 4
                st.rerun()

# Step 4: Save and Analysis
else:
    st.header("Step 4: Calibration Complete")
    
    # Combine all data
    all_data = st.session_state.phase1_data + st.session_state.phase2_data + st.session_state.phase3_data
    
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Display summary statistics
        st.subheader("📊 Calibration Summary")
        
        summary_cols = st.columns(3)
        
        with summary_cols[0]:
            normal_data = df[df['label'] == 0]
            if not normal_data.empty:
                st.metric("Normal Blinks", normal_data['total_blinks'].iloc[-1])
                st.metric("Normal Avg EAR", f"{normal_data['ear_avg'].mean():.3f}")
        
        with summary_cols[1]:
            strain_data = df[df['label'] == 1]
            if not strain_data.empty:
                st.metric("Strain Simulation Blinks", strain_data['total_blinks'].iloc[-1])
                st.metric("Strain Avg EAR", f"{strain_data['ear_avg'].mean():.3f}")
        
        with summary_cols[2]:
            rapid_data = df[df['label'] == 2]
            if not rapid_data.empty:
                st.metric("Rapid Blinks", rapid_data['total_blinks'].iloc[-1])
                st.metric("Rapid Avg EAR", f"{rapid_data['ear_avg'].mean():.3f}")
        
        # Visualizations
        st.subheader("📈 EAR Patterns")
        
        # Create separate DataFrames for plotting
        plot_data = pd.DataFrame()
        for label, name in [(0, 'Normal'), (1, 'Strain'), (2, 'Rapid')]:
            subset = df[df['label'] == label].copy()
            if not subset.empty:
                subset['Phase'] = name
                plot_data = pd.concat([plot_data, subset])
        
        if not plot_data.empty:
            # EAR over time for each phase
            chart_data = plot_data.pivot_table(
                index='timestamp', 
                columns='Phase', 
                values='ear_avg', 
                aggfunc='first'
            ).fillna(method='ffill')
            
            st.line_chart(chart_data)
            
            # Blink detection visualization
            st.subheader("👁️ Blink Detection Analysis")
            blink_summary = plot_data.groupby('Phase').agg({
                'blink_detected': 'sum',
                'ear_avg': 'mean',
                'fixation_duration': 'mean',
                'saccade_count': 'max'
            }).round(3)
            
            st.dataframe(blink_summary)
        
        # Save calibration data
        if st.button("💾 Save Calibration Data", type="primary"):
            # Save to CSV
            calibration_file = os.path.join(user_folder, "enhanced_calibration.csv")
            df.to_csv(calibration_file, index=False)
            
            # Calculate and save thresholds
            normal_ear = df[df['label'] == 0]['ear_avg'].mean()
            strain_ear = df[df['label'] == 1]['ear_avg'].mean()
            rapid_ear = df[df['label'] == 2]['ear_avg'].mean()
            
            thresholds = {
                'normal_ear_mean': normal_ear,
                'strain_ear_mean': strain_ear,
                'rapid_ear_mean': rapid_ear,
                'blink_threshold': normal_ear * 0.7,
                'strain_threshold': (normal_ear + strain_ear) / 2,
                'rapid_threshold': (normal_ear + rapid_ear) / 2,
                'calibration_date': datetime.now().isoformat(),
                'user_id': user_id
            }
            
            # Save thresholds
            threshold_file = os.path.join(user_folder, "thresholds.json")
            with open(threshold_file, 'w') as f:
                json.dump(thresholds, f, indent=2)
            
            # Also save legacy format for compatibility
            legacy_file = os.path.join(user_folder, "strain_data.csv")
            legacy_df = df[['timestamp', 'ear_avg', 'total_blinks', 'label']].copy()
            legacy_df.columns = ['timestamp', 'ear', 'blink_count', 'label']
            legacy_df.to_csv(legacy_file, index=False)
            
            st.success("🎉 Calibration data saved successfully!")
            st.info(f"📁 Files saved:\n- {calibration_file}\n- {threshold_file}\n- {legacy_file}")
            
            # Reset for next calibration
            if st.button("🔄 Start New Calibration"):
                for key in ['calibration_step', 'phase1_data', 'phase2_data', 'phase3_data']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
            
            # Link to dashboard
            st.markdown("### 🎯 Next Steps")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Go to Dashboard"):
                    st.switch_page("pages/dashboard.py")
            with col2:
                if st.button("🔴 Start Monitoring"):
                    st.switch_page("pages/enhanced_monitor.py")

# Instructions and Tips
with st.expander("ℹ️ Calibration Tips"):
    st.markdown("""
    ### 🎯 How to Get Best Results:
    
    **For Normal Blinking:**
    - Sit naturally as you would when working
    - Don't try to control your blinking
    - Look at the screen normally
    
    **For Strain Simulation:**
    - Squint slightly or keep eyes partially closed
    - Blink more slowly
    - Try to simulate how your eyes feel when tired
    
    **For Rapid Blinking:**
    - Blink more frequently than normal
    - Simulate dry or irritated eyes
    - Don't force unnatural movements
    
    ### 📊 What We Measure:
    - **EAR (Eye Aspect Ratio)**: How open your eyes are
    - **Blink Rate**: Frequency of blinking
    - **Blink Duration**: How long eyes stay closed
    - **Fixation Patterns**: Eye movement stability
    - **Saccade Count**: Rapid eye movements
    """)

# Debug information (only show if data exists)
if any([st.session_state.phase1_data, st.session_state.phase2_data, st.session_state.phase3_data]):
    with st.expander("🔍 Debug Information"):
        st.write("Session State:")
        st.write(f"Step: {st.session_state.calibration_step}")
        st.write(f"Phase 1 samples: {len(st.session_state.phase1_data)}")
        st.write(f"Phase 2 samples: {len(st.session_state.phase2_data)}")
        st.write(f"Phase 3 samples: {len(st.session_state.phase3_data)}")
        
        if st.session_state.calibration_step > 4 and all_data:
            st.write("Sample data:")
            st.dataframe(df.head())