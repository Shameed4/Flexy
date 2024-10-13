import time
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to generate vertical lines for each shoulder
def generate_vertical_paths(num_points, start_y, end_y, shoulder_x):
    path = []
    for i in range(num_points):
        y = int(start_y + i * (end_y - start_y) / num_points)
        path.append((shoulder_x, y))
    return path

# Function to draw the predefined paths
def draw_paths(image, paths, touched_points_list):
    for path, touched_points in zip(paths, touched_points_list):
        for i, point in enumerate(path):
            color = (0, 0, 255) if not touched_points[i] else (0, 255, 0)
            cv2.circle(image, point, 5, color, -1)

cap = cv2.VideoCapture(1)

# Initialize variables
predefined_paths = []
touched_points_list = []

while True:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width, _ = image.shape

    results_pose = pose.process(image_rgb)

    if results_pose.pose_landmarks:
        landmarks = results_pose.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        left_shoulder_x = int(left_shoulder.x * width)
        left_shoulder_y = int(left_shoulder.y * height)
        right_shoulder_x = int(right_shoulder.x * width)
        right_shoulder_y = int(right_shoulder.y * height)

        # Initialize paths
        if not predefined_paths:
            num_points = 100
            start_y = left_shoulder_y + 50  # Start lower
            end_y = left_shoulder_y - 50    # End higher

            # Paths for left and right shoulders
            left_path = generate_vertical_paths(num_points, start_y, end_y, left_shoulder_x)
            right_path = generate_vertical_paths(num_points, start_y, end_y, right_shoulder_x)

            predefined_paths = [left_path, right_path]
            touched_points_list = [[False] * num_points, [False] * num_points]

        # Draw shoulders
        cv2.circle(image, (left_shoulder_x, left_shoulder_y), 10, (255, 0, 0), -1)
        cv2.circle(image, (right_shoulder_x, right_shoulder_y), 10, (255, 0, 0), -1)

        # Check for left shoulder
        for i, point in enumerate(predefined_paths[0]):
            if not touched_points_list[0][i] and np.linalg.norm(np.array([left_shoulder_x, left_shoulder_y]) - np.array(point)) < 20:
                touched_points_list[0][i] = True

        # Check for right shoulder
        for i, point in enumerate(predefined_paths[1]):
            if not touched_points_list[1][i] and np.linalg.norm(np.array([right_shoulder_x, right_shoulder_y]) - np.array(point)) < 20:
                touched_points_list[1][i] = True

    # Draw paths
    draw_paths(image, predefined_paths, touched_points_list)

    # Check if all points have been touched
    if all(touched_points_list[0]) and all(touched_points_list[1]):
        break

    cv2.imshow("Shoulder Exercise - Shrugs", image)

    key = cv2.waitKey(5) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Exercise completed at:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
