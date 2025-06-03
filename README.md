# 🏋️‍♂️ Gym-Trainer: AI-Based Rep Counter with Posture Validation

**Gym-Trainer** is a computer vision-based fitness assistant that tracks workout repetitions in real time — but with a twist: **it only counts reps when your posture is correct.**  
Powered by OpenCV and pose estimation techniques, it brings smart feedback to your home workouts.

---

## 📌 Key Features

- ✅ **Posture Validation:** Ensures proper form before counting any rep.
- 🔄 **Real-Time Feedback:** Uses webcam to track your exercise movements.
- 🔢 **Automatic Rep Counting:** No need for manual input — reps are detected live.
- 🧠 **Pose Estimation:** Built on MediaPipe or OpenPose (depending on your implementation).
- 📈 **Accurate Tracking:** Reduces false counts and improves training discipline.

---

## 🖥️ How It Works

1. The webcam captures your movement.
2. The app uses pose detection to identify key points on your body.
3. It analyzes your posture in real time.
4. A rep is **only counted** if the required body posture is correct.
5. Feedback is shown live via the display window.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/MELVIN10101/Gym-Trainer.git
cd Gym-Trainer
```
### 2. Run the Application
```bash
python main.py
```

