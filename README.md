# 👁️ Facial Emotion and Eye Strain Monitoring System

A deep learning–powered fatigue monitoring system that uses only a **standard webcam** to detect both **digital eye strain** and **facial emotions** in real time. This dual-CNN architecture ensures accurate, adaptive, and user-specific fatigue alerts.

---

### 📁 Repository Structure

```
.
├── pages/                  # Additional Streamlit pages (modular views)
├── user_data/              # Contains user-specific models and data
│   └── [user_id]/          
│       ├── strain_data.csv
│       ├── eye_model.h5
│       └── logs/
├── app.py                  # Main entry point of the Streamlit app
├── face_model.h5           # Pretrained facial emotion recognition model
├── user_profiles.csv       # Stores user login info and metadata
├── requirements.txt        # All required Python packages
└── README.md               # This file
```

## ⚙️ How It Works

1. **Eye Strain Detection**:
   - Personalized calibration for each user (3 blink states)
   - Eye features like EAR, blink frequency, gaze variation
   - CNN model trained and saved as `eye_model.h5`

2. **Facial Emotion Recognition**:
   - Based on EfficientNet-B2 (pre-trained)
   - Detects `Neutral`, `Tired`, and `Stressed` expressions
   - Model saved as `face_model.h5`

3. **Fusion Logic**:
   - Fatigue alert is raised **only when both eye and emotion indicators align**
   - Reduces false positives and increases reliability

---
## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/fatigue-monitoring-system.git
cd fatigue-monitoring-system
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the App
```bash
streamlit run app.py
```
---
## ✅ Workflow

1. Create account / login  
2. Complete blink calibration (guided: normal, low, rapid)  
3. Start live monitoring  
4. View session analytics

---

## 📦 Requirements

All dependencies are listed in `requirements.txt`.  
Main libraries include:

- `streamlit`  
- `tensorflow`  
- `opencv-python`  
- `mediapipe`  
- `pandas`, `numpy`, etc.

