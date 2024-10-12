import time
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# Function to generate an oval path with a gap
def generate_oval_path_with_gap(num_points, radius_x, radius_y, center, gap_start_ratio, gap_length_ratio):
    path = []
    gap_start = int(gap_start_ratio * num_points)
    gap_end = gap_start + int(gap_length_ratio * num_points)

    for i in range(num_points):
        if gap_start <= i < gap_end:
            continue  # Skip points in the gap range

        angle = (i / num_points) * 2 * np.pi  # Full ellipse
        x = int(center[0] + radius_x * np.cos(angle))
        y = int(center[1] + radius_y * np.sin(angle))
        path.append((x, y))
    return path


# Function to draw the predefined path with start and end points and gap
def draw_path(image, path, touched_points, start_gap, end_gap):
    for i, point in enumerate(path):
        # Handle the starting and ending circle colors
        if i == start_gap:
            color = (153, 255, 255)  # Pastel yellow for the starting circle
        elif i == len(path) - end_gap - 1:
            color = (153, 153, 255)  # Pastel red for the ending circle
        elif i < start_gap or i >= len(path) - end_gap:
            continue  # Skip drawing the gap circles
        else:
            color = (0, 255, 255) if touched_points[i] else (0, 0, 0)  # Yellow when touched, black when not
        cv2.circle(image, point, 10, color, -1)  # Draw the path with small circles

# Function to add the banner from an existing image with aspect ratio preservation
def add_banner_from_image(image, banner_image):
    # Get the width of the frame
    frame_width = image.shape[1]

    # Resize the banner while maintaining aspect ratio
    banner_aspect_ratio = banner_image.shape[1] / banner_image.shape[0]
    new_banner_width = frame_width
    new_banner_height = int(frame_width / banner_aspect_ratio)

    # Resize the banner to fit the width of the frame
    banner_resized = cv2.resize(banner_image, (new_banner_width, new_banner_height))

    # Add padding if the resized banner height is less than the expected banner height
    expected_banner_height = 400  # Expected banner height
    if banner_resized.shape[0] < expected_banner_height:
        padding = np.zeros((expected_banner_height - banner_resized.shape[0], new_banner_width, 3), dtype=np.uint8)
        banner_resized = np.vstack((banner_resized, padding))

    # Concatenate the banner with the image vertically
    full_image = np.vstack((banner_resized, image))
    return full_image


# Load the existing banner image (adjust the path to your banner image)
banner_image = cv2.imread('ArthritEase.png')


cap = cv2.VideoCapture(1)  # Change to the appropriate camera index

# Initialize variables
predefined_paths = []
touched_points = []

# Define variables for checking the last n circles touched
last_n_circles = 4  # Number of circles at the end that must be touched before quitting
current_path = 0  # Since we are only using one oval path

# Gap settings for start and end
start_gap = 10  # Number of points at the start that will be skipped (indicating the gap)
end_gap = 10  # Number of points at the end that will be skipped (indicating the gap)

while True:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width, _ = image.shape
    center = (width // 2, height // 2)

    # Increase the radius values to make the oval larger and clearer
    max_radius = min(width, height) // 3  # Increase the radius for better visibility
    radius_x = max_radius  # Width of the oval
    radius_y = max_radius * 0.75  # Height of the oval (adjusted to maintain proper proportions)

    if not predefined_paths:
        num_points = 200
        predefined_paths.append(
            generate_oval_path_with_gap(num_points, radius_x, radius_y, center, 0.05, 0.10))  # Adjusted gap ratios
        touched_points.append([False] * len(predefined_paths[0]))

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
    draw_path(image, predefined_paths[0], touched_points[0], start_gap, end_gap)
    # Add the banner from the image at the top
    full_image = add_banner_from_image(image, banner_image)

    # Check if the last 'n' circles have been touched
    if all(touched_points[current_path][-last_n_circles - end_gap:-end_gap]):
        # Calculate the percentage of circles touched
        total_circles = len(touched_points[current_path]) - start_gap - end_gap
        touched_circles = sum(touched_points[current_path][start_gap:-end_gap])
        percentage_touched = (touched_circles / total_circles) * 100

        # Show the percentage of touched circles
        print(f"{percentage_touched:.2f}")
        break

    cv2.imshow("Neck Exercise - Oval", full_image)

    key = cv2.waitKey(5) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
