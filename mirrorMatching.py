import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load a target pose image (silhouette)
target_image = cv2.imread('target_pose.png', cv2.IMREAD_UNCHANGED)
target_image = cv2.resize(target_image, (640, 480))

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Mirror image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = frame.shape

    results = pose.process(frame_rgb)

    # Overlay the target pose on the frame
    overlay = frame.copy()
    alpha = 0.5  # Transparency factor
    cv2.addWeighted(target_image, alpha, overlay, 1 - alpha, 0, overlay)
    frame = overlay

    if results.pose_landmarks:
        # Draw user's pose landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style())

        # Check if the user's pose matches the target pose
        # For simplicity, we'll check if certain key landmarks are within a threshold
        landmarks = results.pose_landmarks.landmark

        # Define the target positions for specific landmarks (normalized coordinates)
        target_landmarks = {
            mp_pose.PoseLandmark.LEFT_WRIST: (0.3, 0.5),
            mp_pose.PoseLandmark.RIGHT_WRIST: (0.7, 0.5),
            mp_pose.PoseLandmark.LEFT_ELBOW: (0.4, 0.4),
            mp_pose.PoseLandmark.RIGHT_ELBOW: (0.6, 0.4)
        }

        match = True
        threshold = 0.1  # Acceptable deviation

        for landmark_id, target_pos in target_landmarks.items():
            user_landmark = landmarks[landmark_id]
            distance = np.linalg.norm(np.array([user_landmark.x, user_landmark.y]) - np.array(target_pos))
            if distance > threshold:
                match = False
                break

        if match:
            cv2.putText(frame, "Pose Matched!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Mirror Motion Matching", frame)
            cv2.waitKey(2000)
            break

    cv2.imshow("Mirror Motion Matching", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
