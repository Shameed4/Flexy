import time
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to generate circular paths for each elbow
def generate_elbow_circle_paths(num_points, radius, elbow_center):
    path = []
    for i in range(num_points):
        angle = (i / num_points) * 2 * np.pi  # Full circle
        x = int(elbow_center[0] + radius * np.cos(angle))
        y = int(elbow_center[1] + radius * np.sin(angle))
        path.append((x, y))
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
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]

        left_elbow_x = int(left_elbow.x * width)
        left_elbow_y = int(left_elbow.y * height)
        right_elbow_x = int(right_elbow.x * width)
        right_elbow_y = int(right_elbow.y * height)

        # Initialize paths
        if not predefined_paths:
            num_points = 100
            radius = 50  # Adjust radius as needed

            left_path = generate_elbow_circle_paths(num_points, radius, (left_elbow_x, left_elbow_y))
            right_path = generate_elbow_circle_paths(num_points, radius, (right_elbow_x, right_elbow_y))

            predefined_paths = [left_path, right_path]
            touched_points_list = [[False] * num_points, [False] * num_points]

        # Draw elbows
        cv2.circle(image, (left_elbow_x, left_elbow_y), 10, (255, 0, 0), -1)
        cv2.circle(image, (right_elbow_x, right_elbow_y), 10, (255, 0, 0), -1)

        # Check for left elbow
        for i, point in enumerate(predefined_paths[0]):
            if not touched_points_list[0][i] and np.linalg.norm(np.array([left_elbow_x, left_elbow_y]) - np.array(point)) < 20:
                touched_points_list[0][i] = True

        # Check for right elbow
        for i, point in enumerate(predefined_paths[1]):
            if not touched_points_list[1][i] and np.linalg.norm(np.array([right_elbow_x, right_elbow_y]) - np.array(point)) < 20:
                touched_points_list[1][i] = True

    # Draw paths
    draw_paths(image, predefined_paths, touched_points_list)

    # Check if all points have been touched
    if all(touched_points_list[0]) and all(touched_points_list[1]):
        break

    cv2.imshow("Shoulder Exercise - Circles", image)

    key = cv2.waitKey(5) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Exercise completed at:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
