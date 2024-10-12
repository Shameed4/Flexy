import time
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to generate an oval path
def generate_oval_path(num_points, radius_x, radius_y, center):
    path = []
    for i in range(num_points):
        angle = (i / num_points) * 2 * np.pi  # Full ellipse
        x = int(center[0] + radius_x * np.cos(angle))
        y = int(center[1] + radius_y * np.sin(angle))
        path.append((x, y))
    return path


# Function to draw the predefined path
def draw_path(image, path, touched_points):
    for i, point in enumerate(path):
        color = (0, 0, 255) if not touched_points[i] else (0, 255, 0)
        cv2.circle(image, point, 5, color, -1)

cap = cv2.VideoCapture(0)  # Change to the appropriate camera index

# Initialize variables
predefined_paths = []
touched_points = []

while True:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width, _ = image.shape
    center = (width // 2, height // 2)
    max_radius = min(width, height) // 4

    if not predefined_paths:
        num_points = 200
        radius_x = max_radius  # Width of the oval
        radius_y = max_radius // 2  # Height of the oval (adjust as desired)
        predefined_paths.append(generate_oval_path(num_points, radius_x, radius_y, center))
        touched_points.append([False] * num_points)


    results_pose = pose.process(image_rgb)

    if results_pose.pose_landmarks:
        # Get the nose position
        nose = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        nose_x = int(nose.x * width)
        nose_y = int(nose.y * height)

        # Draw the nose position
        cv2.circle(image, (nose_x, nose_y), 10, (255, 0, 0), -1)

        # Check if nose is near predefined path
        for i, point in enumerate(predefined_paths[0]):
            if not touched_points[0][i] and np.linalg.norm(np.array([nose_x, nose_y]) - np.array(point)) < 20:
                touched_points[0][i] = True

    # Draw the path
    draw_path(image, predefined_paths[0], touched_points[0])

    # Check if all points have been touched
    if all(touched_points[0]):
        break

    cv2.imshow("Neck Exercise - Circle", image)

    key = cv2.waitKey(5) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Exercise completed at:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))