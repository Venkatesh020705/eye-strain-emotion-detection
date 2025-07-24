import streamlit as st
import os
import pandas as pd
import json
from glob import glob
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ----------------------------
# Setup
st.set_page_config(page_title="Enhanced Dashboard", layout="wide")
st.title("📊 Enhanced Eye Strain Monitoring Dashboard")

user_id = st.session_state.get("user_id")
user_folder = st.session_state.get("user_folder")
if not user_id or not user_folder:
    st.error("User not logged in. Please go back to home.")
    st.stop()

st.success(f"Logged in as: {user_id}")

# ----------------------------
# Load user calibration data
calibration_file = os.path.join(user_folder, "enhanced_calibration.csv")
thresholds_file = os.path.join(user_folder, "thresholds.json")

if os.path.exists(thresholds_file):
    with open(thresholds_file, 'r') as f:
        thresholds = json.load(f)
    
    st.markdown("### 🎯 Your Calibrated Thresholds")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Normal EAR", f"{thresholds.get('normal_ear_mean', 0):.3f}")
    with col2:
        st.metric("Blink Threshold", f"{thresholds.get('blink_threshold', 0):.3f}")
    with col3:
        st.metric("Strain Threshold", f"{thresholds.get('strain_threshold', 0):.3f}")
    with col4:
        st.metric("Calibration Date", thresholds.get('calibration_date', 'Unknown')[:10])

# ----------------------------
# Load recent session data
sessions = sorted(glob(os.path.join(user_folder, "session_*.csv")), reverse=True)

if sessions:
    st.markdown("### 📁 Recent Sessions")
    
    # Session selector
    selected_session = st.selectbox(
        "Select a session to analyze:",
        sessions,
        format_func=lambda x: f"Session {os.path.basename(x).replace('session_', '').replace('.csv', '')}"
    )
    
    # Load and display session data
    df = pd.read_csv(selected_session)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Session summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        duration = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60
        st.metric("Session Duration", f"{duration:.1f} min")
    
    with col2:
        if 'blink_count' in df.columns:
            total_blinks = df['blink_count'].max()
            blink_rate = total_blinks / duration if duration > 0 else 0
            st.metric("Total Blinks", f"{total_blinks} ({blink_rate:.1f}/min)")
    
    with col3:
        if 'emotion' in df.columns:
            most_common_emotion = df['emotion'].mode()[0] if len(df) > 0 else "N/A"
            st.metric("Dominant Emotion", most_common_emotion)
    
    with col4:
        if 'strain_alerts' in df.columns:
            alerts = df['strain_alerts'].max()
            st.metric("Strain Alerts", alerts)
    
    # Visualizations
    st.markdown("### 📈 Session Analysis")
    
    # EAR over time
    if 'ear' in df.columns:
        fig_ear = px.line(df, x='timestamp', y='ear', title='Eye Aspect Ratio Over Time')
        if os.path.exists(thresholds_file):
            # Add threshold lines
            fig_ear.add_hline(y=thresholds.get('blink_threshold', 0.25), 
                             line_dash="dash", line_color="red", 
                             annotation_text="Blink Threshold")
            fig_ear.add_hline(y=thresholds.get('normal_ear_mean', 0.3), 
                             line_dash="dash", line_color="green", 
                             annotation_text="Normal EAR")
        st.plotly_chart(fig_ear, use_container_width=True)
    
    # Emotion distribution (if available)
    if 'emotion' in df.columns:
        emotion_counts = df['emotion'].value_counts()
        fig_emotions = px.pie(values=emotion_counts.values, names=emotion_counts.index,
                             title='Emotion Distribution')
        st.plotly_chart(fig_emotions, use_container_width=True)
    
    # Blink rate analysis
    if 'blink_count' in df.columns:
        # Calculate blink rate in 1-minute windows
        df_resampled = df.set_index('timestamp').resample('1T')['blink_count'].max().diff().fillna(0)
        fig_blinks = px.line(x=df_resampled.index, y=df_resampled.values,
                            title='Blink Rate (Blinks per Minute)')
        st.plotly_chart(fig_blinks, use_container_width=True)
else:
    st.info("No session data found. Please complete calibration and run monitoring first.")

# ----------------------------
# Calibration data analysis
if os.path.exists(calibration_file):
    st.markdown("### 🔬 Calibration Analysis")
    
    calib_df = pd.read_csv(calibration_file)
    
    # EAR comparison across phases
    phase_names = {0: 'Normal', 1: 'Strain', 2: 'Rapid'}
    calib_df['phase'] = calib_df['label'].map(phase_names)
    
    fig_calib = px.box(calib_df, x='phase', y='ear_avg', 
                       title='EAR Distribution by Calibration Phase')
    st.plotly_chart(fig_calib, use_container_width=True)
    
    # Blink detection analysis
    blink_analysis = calib_df.groupby('phase').agg({
        'blink_detected': 'sum',
        'total_blinks': 'max',
        'ear_avg': 'mean',
        'fixation_duration': 'mean'
    }).round(3)
    
    st.dataframe(blink_analysis, use_container_width=True)

# ----------------------------
# Historical trends
st.markdown("")

if len(sessions) >= 1:
    # Load all sessions for trend analysis
    all_sessions = []
    for session_file in sessions[:10]:  # Last 10 sessions
        try:
            session_df = pd.read_csv(session_file)
            session_name = os.path.basename(session_file).replace('session_', '').replace('.csv', '')
            session_df['session'] = session_name
            all_sessions.append(session_df)
        except:
            continue
    
    if all_sessions:
        combined_df = pd.concat(all_sessions, ignore_index=True)
        
        # Session-level metrics
        session_metrics = combined_df.groupby('session').agg({
            'ear': 'mean',
            'blink_count': 'max',
            'strain_alerts': 'max' if 'strain_alerts' in combined_df.columns else 'count'
        }).reset_index()
        
        # Trends
        col1, col2 = st.columns(2)
        
        with col1:
            fig_ear_trend = px.line(session_metrics, x='session', y='ear',
                                   title='Average EAR Trend')
            st.plotly_chart(fig_ear_trend, use_container_width=True)
        
        with col2:
            fig_blink_trend = px.bar(session_metrics, x='session', y='blink_count',
                                    title='Blinks per Session')
            st.plotly_chart(fig_blink_trend, use_container_width=True)

# ----------------------------
# Quick actions
st.markdown("### 🔁 Quick Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔴 Start Monitoring", type="primary"):
        st.switch_page("pages/enhanced_monitor.py")

with col2:
    if st.button("🔄 Recalibrate"):
        st.switch_page("pages/enhanced_calibration.py")

with col3:
    if st.button("📁 Export Data"):
        # Create combined export
        export_data = {}
        
        if os.path.exists(calibration_file):
            export_data['calibration'] = pd.read_csv(calibration_file)
        
        if sessions:
            latest_session = pd.read_csv(sessions[0])
            export_data['latest_session'] = latest_session
        
        if os.path.exists(thresholds_file):
            with open(thresholds_file, 'r') as f:
                export_data['thresholds'] = json.load(f)
        
        # For demo purposes, just show success message
        st.success("Data export ready! (Implementation would download files)")

with col4:
    if st.button("🏠 Home"):
        st.switch_page("app.py")

# ----------------------------
# Settings and preferences
with st.expander("⚙️ Settings & Preferences"):
    st.markdown("### User Preferences")
    
    # Notification settings
    enable_notifications = st.checkbox("Enable strain notifications", True)
    notification_frequency = st.slider("Alert frequency (minutes)", 5, 60, 20)
    
    # Threshold adjustments
    if os.path.exists(thresholds_file):
        st.markdown("### Threshold Adjustments")
        st.info("You can fine-tune your thresholds based on your experience.")
        
        current_blink = thresholds.get('blink_threshold', 0.25)
        new_blink = st.slider("Blink detection threshold", 0.15, 0.35, current_blink, 0.01)
        
        current_strain = thresholds.get('strain_threshold', 0.22)
        new_strain = st.slider("Strain detection threshold", 0.15, 0.35, current_strain, 0.01)
        
        if st.button("💾 Save Threshold Changes"):
            thresholds['blink_threshold'] = new_blink
            thresholds['strain_threshold'] = new_strain
            thresholds['last_modified'] = datetime.now().isoformat()
            
            with open(thresholds_file, 'w') as f:
                json.dump(thresholds, f, indent=2)
            
            st.success("Thresholds updated successfully!")
            st.rerun()
    
    # Data management
    st.markdown("### Data Management")
    if st.button("🗑️ Clear All Data", help="This will delete all your calibration and session data"):
        # For safety, require confirmation
        if st.button("⚠️ Confirm Delete (This cannot be undone)"):
            # Delete all user data
            import shutil
            shutil.rmtree(user_folder)
            st.success("All data cleared. Please recalibrate.")
            st.rerun()