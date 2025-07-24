import streamlit as st
import os
import pandas as pd
from datetime import datetime

# ----------------------------
# Paths
USER_DIR = "user_data"
PROFILE_CSV = "user_profiles.csv"

# Ensure required folders/files exist
os.makedirs(USER_DIR, exist_ok=True)
if not os.path.exists(PROFILE_CSV) or os.stat(PROFILE_CSV).st_size == 0:
    pd.DataFrame(columns=["user_id", "created_at", "last_session"]).to_csv(PROFILE_CSV, index=False)

# ----------------------------
# Page Config
st.set_page_config(page_title="Enhanced Eye Strain App", layout="centered", page_icon="👁️")
st.title("👁️ Enhanced Eye Strain Monitor - Login")

# ----------------------------
# Session Logic
if "user_id" not in st.session_state:
    st.markdown("""
    ### 🔍 What's New in This Enhanced Version:
    - **Improved Blink Detection** with auto-calibration
    - **Emotion Recognition** from facial expressions
    - **Better Eye Strain Detection** with personalized thresholds
    - **Real-time Monitoring** with alerts and notifications
    - **Detailed Analytics** and session tracking
    """)
    
    username = st.text_input("Enter your name to start:", placeholder="e.g., john_doe")

    if username:
        username = username.strip().lower().replace(" ", "_")
        user_id = f"user_{username}"
        user_folder = os.path.join(USER_DIR, user_id)
        os.makedirs(user_folder, exist_ok=True)

        # Load or create user profile
        profiles = pd.read_csv(PROFILE_CSV)
        if user_id not in profiles["user_id"].values:
            new_row = pd.DataFrame([[user_id, datetime.now(), "Never"]], columns=profiles.columns)
            profiles = pd.concat([profiles, new_row], ignore_index=True)
            st.success(f"👤 New user created: {user_id}")
        else:
            st.info(f"👋 Welcome back: {user_id}")

        # Update last session
        profiles.loc[profiles.user_id == user_id, "last_session"] = datetime.now()
        profiles.to_csv(PROFILE_CSV, index=False)

        # Save session state and rerun
        st.session_state.user_id = user_id
        st.session_state.user_folder = user_folder
        st.rerun()

else:
    user_id = st.session_state.user_id
    user_folder = st.session_state.user_folder
    
    # Check for enhanced calibration first
    enhanced_calibration_path = os.path.join(user_folder, "enhanced_calibration.csv")
    thresholds_path = os.path.join(user_folder, "thresholds.json")
    
    if not os.path.exists(enhanced_calibration_path):
        st.success("🎉 Welcome! Let's set up your personalized eye tracking.")
        st.info("The enhanced calibration will create custom thresholds for accurate detection.")
        if st.button("🚀 Start Enhanced Setup", type="primary"):
            st.switch_page("pages/user_info.py")
    else:
        st.success("✅ Enhanced calibration complete! Ready for monitoring.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 View Dashboard", type="primary"):
                st.switch_page("pages/dashboard.py")
        with col2:
            if st.button("🔴 Start Live Monitoring", type="secondary"):
                st.switch_page("pages/enhanced_monitor.py")
        
        st.markdown("---")
        if st.button("🔄 Recalibrate", help="Run calibration again to update your thresholds"):
            st.switch_page("pages/enhanced_calibration.py")