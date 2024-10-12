import cv2
import mediapipe as mp

# Initialize MediaPipe Pose and Drawing tools
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Landmark names for reference
landmark_names = [
    "Nose", "Left eye (inner)", "Left eye", "Left eye (outer)", "Right eye (inner)", "Right eye",
    "Right eye (outer)", "Left ear", "Right ear", "Mouth (left)", "Mouth (right)", "Left shoulder",
    "Right shoulder", "Left elbow", "Right elbow", "Left wrist", "Right wrist", "Left pinky",
    "Right pinky", "Left index", "Right index", "Left thumb", "Right thumb", "Left hip", "Right hip",
    "Left knee", "Right knee", "Left ankle", "Right ankle", "Left heel", "Right heel",
    "Left foot index", "Right foot index"
]

# Open webcam
cap = cv2.VideoCapture(0)

# Set up Pose estimation model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Convert frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detections
        results = pose.process(image)

        # Convert image back to BGR for display
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Check for "p" key press to print coordinates with names
            if cv2.waitKey(1) & 0xFF == ord('p'):
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    # Extract x, y, z, and visibility
                    x, y, z = landmark.x, landmark.y, landmark.z
                    visibility = landmark.visibility
                    # Print landmark name with coordinates and visibility
                    print(f"{landmark_names[idx]}: x={x:.4f}, y={y:.4f}, z={z:.4f}, visibility={visibility:.4f}")

        # Display output
        cv2.imshow('Pose Detection', image)

        # Exit loop on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
