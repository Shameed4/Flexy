import time
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to generate a line path
def generate_line_path(num_points, start_point, end_point):
    path = []
    for i in range(num_points):
        x = int(start_point[0] + i * (end_point[0] - start_point[0]) / num_points)
        y = int(start_point[1] + i * (end_point[1] - start_point[1]) / num_points)
        path.append((x, y))
    return path

# Function to draw the predefined path
def draw_path(image, path, touched_points):
    for i, point in enumerate(path):
        color = (0, 0, 255) if not touched_points[i] else (0, 255, 0)
        cv2.circle(image, point, 5, color, -1)

cap = cv2.VideoCapture(1)

# Initialize variables
predefined_paths = []
touched_points = []
current_path_index = 0  # 0 for "Yes", 1 for "No"

while True:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width, _ = image.shape

    if not predefined_paths:
        num_points = 200
        # "Yes" movement (vertical line)
        start_point_yes = (width // 2, height // 4)
        end_point_yes = (width // 2, 3 * height // 4)
        predefined_paths.append(generate_line_path(num_points, start_point_yes, end_point_yes))
        touched_points.append([False] * num_points)
        # "No" movement (horizontal line)
        start_point_no = (width // 4, height // 2)
        end_point_no = (3 * width // 4, height // 2)
        predefined_paths.append(generate_line_path(num_points, start_point_no, end_point_no))
        touched_points.append([False] * num_points)

    results_pose = pose.process(image_rgb)

    if results_pose.pose_landmarks:
        nose = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        nose_x = int(nose.x * width)
        nose_y = int(nose.y * height)
        cv2.circle(image, (nose_x, nose_y), 10, (255, 0, 0), -1)

        # Check if nose is near predefined path
        current_path = predefined_paths[current_path_index]
        current_touched = touched_points[current_path_index]
        for i, point in enumerate(current_path):
            if not current_touched[i] and np.linalg.norm(np.array([nose_x, nose_y]) - np.array(point)) < 20:
                current_touched[i] = True

    # Draw the path
    draw_path(image, predefined_paths[current_path_index], touched_points[current_path_index])

    # Check if all points have been touched
    if all(touched_points[current_path_index]):
        if current_path_index == 0:
            # Move to "No" movement after completing "Yes"
            current_path_index = 1
        else:
            # Both movements completed
            break

    cv2.imshow("Neck Exercise - 'Yes' and 'No' Movements", image)

    key = cv2.waitKey(5) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Exercise completed at:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
