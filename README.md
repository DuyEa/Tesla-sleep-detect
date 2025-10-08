# Driver Drowsiness Detection 🚗😴

A personal project inspired by Tesla’s **Full Self-Driving (FSD)** safety features.  
This system detects driver drowsiness in real-time using **computer vision** and **machine learning** to help improve road safety.  

---

## ✨ Features  
- **Eye Aspect Ratio (EAR)** and **Eye Closure Ratio (ECR)** calculation from face landmarks.  
- **Random Forest classifier** trained on eye state features.  
- **Head pose estimation** to detect downward head tilt (sleepy posture).  
- **Live video stream processing** (webcam or video file).  
- **Real-time overlay** with sleep/awake status and visual alerts.  
- Automatic saving of annotated video output.  

---

## 🛠️ Tech Stack  
- **Python 3.9+**  
- **OpenCV** – video stream capture and visualization  
- **Mediapipe** – face mesh landmark extraction  
- **Scikit-learn** – model training and evaluation  
- **Pandas / NumPy** – data preprocessing and feature engineering  

---

## 📂 Project Structure  
```
├── train_sleep.py          # Train Random Forest model on eye dataset  
├── online_detect_ear_ecr.py # Real-time drowsiness detection (webcam/video)  
├── RandFor.pkl             # Saved Random Forest model (generated after training)  
├── Eye dataset/            # Dataset (organized in folders: awake/sleep states)  
└── README.md               # Project documentation  
```  

---

## 🚀 Usage  

### 1. Train the Model  
```bash
python train_sleep.py --root_dir "path/to/Eye dataset" --window 15
```
This will compute EAR/ECR features and train a **Random Forest** classifier.  
The model will be saved as `RandFor.pkl`.  

### 2. Run Real-Time Detection  
Use webcam:  
```bash
python online_detect_ear_ecr.py --camera 0 --model RandFor.pkl
```  

Or run on video file:  
```bash
python online_detect_ear_ecr.py --video driver.mp4 --model RandFor.pkl
```  

Output video with annotated sleep/awake status will be saved automatically (e.g. `driver_detected.mp4`).  

---

## 📊 Example Output  
- **AWAKE:** Green eye outlines, status shown as *AWAKE*.  
- **SLEEP:** Red overlays on eyes + big alert banner *SLEEPY! ALERT!*  

---

## 🔮 Future Work  
- Integrate with car onboard systems for **real-time driver alerts**.  
- Extend to other fatigue signals (yawning, gaze direction).  
- Optimize for **mobile/embedded deployment**.  
