# 👁️ Facial Emotion and Eye Strain Monitoring System

A deep learning–powered fatigue monitoring system that uses only a **standard webcam** to detect both **digital eye strain** and **facial emotions** in real time. This dual-CNN architecture ensures accurate, adaptive, and user-specific fatigue alerts.

---

## 📁 Repository Structure
.
.
├── pages/                 # Additional Streamlit pages (modular views)
├── user_data/             # Contains user-specific models and data
│   └── [user_id]/         
│       ├── strain_data.csv
│       ├── eye_model.h5
│       └── logs/
├── app.py                 # Main entry point of the Streamlit app
├── face_model.h5          # Pretrained facial emotion recognition model
├── user_profiles.csv      # Stores user login info and metadata
├── requirements.txt       # All required Python packages
└── README.md              # This file

---

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

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/fatigue-monitoring-system.git
   cd fatigue-monitoring-system
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the App

bash
Copy
Edit
streamlit run app.py
Workflow

Create account / login

Complete blink calibration (guided)

Start live monitoring

View session analytics

📦 Requirements
All dependencies are listed in requirements.txt. Main libraries include:

streamlit

tensorflow

opencv-python

mediapipe

pandas, numpy, etc.

📊 Performance
Module	Accuracy
Eye Strain Model	95.68%
Emotion Classifier	92.00%
Fusion Model	98.90%

📈 Output
Live fatigue alert (based on dual-model logic)

Blink trends & fatigue classification

Emotion timeline visualization

🧑‍💻 Contributors
Adapala Rishi Manikanta

Baba Ameer Shaik

Yaswanth Kancharla

Venkateswara Reddy Tegulapalle

Guide: Dr. Jyotsna C

📄 License
Licensed under the MIT License.
