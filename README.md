🏋️‍♂️ Gym-Trainer: AI-Based Rep Counter with Posture Validation
Gym-Trainer is a computer vision-based fitness assistant that tracks workout repetitions in real time — but with a twist: it only counts reps when your posture is correct.
Powered by OpenCV and pose estimation techniques, it brings smart feedback to your home workouts.

📌 Key Features
✅ Posture Validation: Ensures proper form before counting any rep.

🔄 Real-Time Feedback: Uses webcam to track your exercise movements.

🔢 Automatic Rep Counting: No need for manual input — reps are detected live.

🧠 Pose Estimation: Built on MediaPipe or OpenPose (depending on your implementation).

📈 Accurate Tracking: Reduces false counts and improves training discipline.

🖥️ How It Works
The webcam captures your movement.

The app uses pose detection to identify key points on your body.

It analyzes your posture in real time.

A rep is only counted if the required body posture is correct.

Feedback is shown live via the display window.

🛠️ Installation & Setup
1. Clone the Repository
bash
Copy code
git clone https://github.com/MELVIN10101/Gym-Trainer.git
cd Gym-Trainer
2. Install Requirements
bash
Copy code
pip install -r requirements.txt
3. Run the Application
bash
Copy code
python main.py
Ensure your webcam is connected and permissions are granted.

