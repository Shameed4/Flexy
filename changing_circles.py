import time

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands and Pose
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to generate circular points
def generate_circle_path(num_points, radius, center):
    return [
        (int(center[0] + radius * np.cos(2 * np.pi * i / num_points)),
         int(center[1] + radius * np.sin(2 * np.pi * i / num_points)))
        for i in range(num_points)
    ]

# Function to generate square points
def generate_square_path(size, center):
    return [
        (center[0] - size // 2, center[1] - size // 2),
        (center[0] + size // 2, center[1] - size // 2),
        (center[0] + size // 2, center[1] + size // 2),
        (center[0] - size // 2, center[1] + size // 2)
    ]

# List to hold all predefined paths and their touched statuses
predefined_paths = []
touched_points = []

# Define paths
predefined_paths.append(generate_circle_path(8, 200, (320, 240)))  # Circle path
touched_points.append([False] * 8)

predefined_paths.append(generate_square_path(200, (320, 240)))  # Square path
touched_points.append([False] * 4)

# Add more paths as needed
predefined_paths.append(generate_circle_path(12, 150, (320, 240)))  # Another circle path
touched_points.append([False] * 12)

# Initialize variables
current_path = 0  # Index of the current path

# Define body part labels for all 33 landmarks
body_part_labels = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
    "Left Pinky", "Right Pinky", "Left Ring", "Right Ring",
    "Left Middle", "Right Middle", "Left Index", "Right Index",
    "Left Thumb", "Right Thumb", "Left Heel", "Right Heel",
    "Left Foot Index", "Right Foot Index", "Left Big Toe", "Right Big Toe",
    "Left Small Toe", "Right Small Toe"
]

# Function to draw the predefined path
def draw_path(image, path, touched_points):
    for i, point in enumerate(path):
        color = (0, 0, 255) if touched_points[i] else (0, 255, 0)
        cv2.circle(image, point, 10, color, -1)

# Function to draw pose landmarks
def draw_pose_landmarks(image, landmarks):
    for i, landmark in enumerate(landmarks.landmark):
        if i < len(body_part_labels):  # Ensure we don't exceed label index
            h, w, _ = image.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Draw landmark
            # Label the landmark with the corresponding body part
            cv2.putText(image, body_part_labels[i], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

cap = cv2.VideoCapture(1)  # Change to the appropriate camera index

while True:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)  # Flip the image for a mirror effect
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(image_rgb)
    results_pose = pose.process(image_rgb)

    # If hands are detected
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            # Get the position of the index finger tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            finger_x = int(index_finger_tip.x * image.shape[1])
            finger_y = int(index_finger_tip.y * image.shape[0])

            # Draw the finger position
            cv2.circle(image, (finger_x, finger_y), 10, (0, 0, 255), -1)

            # Check if finger is near predefined path (simple distance check)
            for i, path_point in enumerate(predefined_paths[current_path]):
                if not touched_points[current_path][i] and np.linalg.norm(np.array([finger_x, finger_y]) - np.array(path_point)) < 20:
                    touched_points[current_path][i] = True

    # If pose landmarks are detected
    if results_pose.pose_landmarks:
        draw_pose_landmarks(image, results_pose.pose_landmarks)

    # Draw the current predefined path, updating colors based on touched status
    draw_path(image, predefined_paths[current_path], touched_points[current_path])

    # Check if all points in the current path have been touched
    if all(touched_points[current_path]):
        print("All points touched! Closing application...")
        break

    cv2.imshow("Air Drawing", image)

    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # Press 'Esc' to exit
        break
    elif key == ord('a'):  # Press 'a' for left (to switch to the previous path)
        current_path = (current_path - 1) % len(predefined_paths)  # Go to the previous path
        touched_points[current_path] = [False] * len(touched_points[current_path])  # Reset touched status
    elif key == ord('d'):  # Press 'd' for right (to switch to the next path)
        current_path = (current_path + 1) % len(predefined_paths)  # Go to the next path
        touched_points[current_path] = [False] * len(touched_points[current_path])  # Reset touched status

cap.release()
cv2.destroyAllWindows()
print(time.time().__str__())