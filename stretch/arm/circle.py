import time
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)


# Function to generate circular points with a high number of points for a smoother look
def generate_detailed_circle_path(num_points, radius, center):
    return [
        (int(center[0] + radius * np.cos(2 * np.pi * i / num_points)),
         int(center[1] + radius * np.sin(2 * np.pi * i / num_points)))
        for i in range(num_points)
    ]


# Function to draw the predefined path
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
        cv2.circle(image, point, 10, color, -1)  # Small circles for the path


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
banner_image = cv2.imread('2.png')

cap = cv2.VideoCapture(1)  # Change to the appropriate camera index

# Initialize variables
current_path = 0  # Index of the current path
predefined_paths = []  # List to hold all predefined paths
touched_points = []  # List to hold touched status for each path
last_n_circles = 4  # Number of circles at the end that must be touched before quitting

# Gap settings for start and end
start_gap = 7  # Number of points at the start that will be skipped (indicating the gap)
end_gap = 7  # Number of points at the end that will be skipped (indicating the gap)

while True:
    success, image = cap.read()
    if not success:
        break

    # Add the banner at the top before modifying the rest of the image
    image = cv2.flip(image, 1)  # Flip the image for a mirror effect
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get the dimensions of the frame (without the banner)
    height, width, _ = image.shape
    banner_height = 150  # Adjust this based on the banner's height
    working_height = height - banner_height  # Adjust for banner space
    center = (width // 2, banner_height + working_height // 2 - 50)  # Adjusting the circle upward by 50 pixels
  # Center below the banner

    # Define the radius dynamically based on the smaller dimension of the working area
    max_radius = min(width, working_height) // 3  # Reduce a bit to leave space

    if not predefined_paths:
        # Generate a circle path with many points (more points = smoother line)
        num_points = 200  # Increase the number of points to make it look like a continuous line
        predefined_paths.append(generate_detailed_circle_path(num_points, max_radius, center))
        touched_points.append([False] * num_points)

    results_hands = hands.process(image_rgb)

    # If hands are detected
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            # Get the position of the index finger tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            finger_x = int(index_finger_tip.x * image.shape[1])
            finger_y = int(index_finger_tip.y * image.shape[0])

            # Draw the finger position
            cv2.circle(image, (finger_x, finger_y), 30, (0, 0, 255), -1)

            # Check if finger is near predefined path (simple distance check)
            for i, path_point in enumerate(predefined_paths[current_path]):
                if start_gap <= i < len(predefined_paths[current_path]) - end_gap:
                    if not touched_points[current_path][i] and np.linalg.norm(
                            np.array([finger_x, finger_y]) - np.array(path_point)) < 20:
                        touched_points[current_path][i] = True

    # Draw the current predefined path, updating colors based on touched status and start/end gaps
    draw_path(image, predefined_paths[current_path], touched_points[current_path], start_gap, end_gap)

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

    # Display the image with the banner
    cv2.imshow("Air Drawing", full_image)

    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # Press 'Esc' to exit manually
        break
    elif key == ord('a'):  # Press 'a' for left (to switch to the previous path)
        current_path = (current_path - 1) % len(predefined_paths)  # Go to the previous path
        touched_points[current_path] = [False] * len(touched_points[current_path])  # Reset touched status
    elif key == ord('d'):  # Press 'd' for right (to switch to the next path)
        current_path = (current_path + 1) % len(predefined_paths)  # Go to the next path
        touched_points[current_path] = [False] * len(touched_points[current_path])  # Reset touched status

cap.release()
cv2.destroyAllWindows()