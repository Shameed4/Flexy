import time
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Function to generate a wave-shaped path with multiple waves
def generate_wave_path(num_points, amplitude, wavelength, center, num_waves):
    path = []
    for i in range(num_points):
        x = int(center[0] + (i - num_points // 2) * (wavelength / num_points))  # Horizontal linear movement
        y = int(center[1] + amplitude * np.sin(2 * np.pi * num_waves * i / num_points))  # Multiple vertical waves
        path.append((x, y))
    return path

# Function to draw the predefined path
def draw_path(image, path, touched_points, start_gap, end_gap):
    for i, point in enumerate(path):
        # Handle the starting and ending circle colors
        if i == start_gap:
            color = (255, 255, 153)  # Pastel yellow for the starting circle
        elif i == len(path) - end_gap - 1:
            color = (255, 153, 153)  # Pastel red for the ending circle
        elif i < start_gap or i >= len(path) - end_gap:
            continue  # Skip drawing the gap circles
        else:
            color = (0, 255, 255) if touched_points[i] else (0, 0, 0)  # Yellow when touched, black when not
        cv2.circle(image, point, 10, color, -1)  # Small circles for the path

cap = cv2.VideoCapture(1)  # Change to the appropriate camera index

# Initialize variables
current_path = 0  # Index of the current path
predefined_paths = []  # List to hold all predefined paths
touched_points = []  # List to hold touched status for each path

# Gap settings for start and end
start_gap = 7  # Number of points at the start that will be skipped (indicating the gap)
end_gap = 7    # Number of points at the end that will be skipped (indicating the gap)
num_ending_circles = 5  # Number of ending circles we care about

while True:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)  # Flip the image for a mirror effect
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get the dimensions of the frame
    height, width, _ = image.shape
    center = (width // 2, height // 2)

    # Define wave parameters
    amplitude = height // 4  # Vertical height of the wave
    wavelength = width // 2  # Horizontal stretch of the wave
    num_points = 200  # Higher number of points for smooth wave shape
    num_waves = 3  # Number of complete waves

    if not predefined_paths:
        # Generate a wave-shaped path with multiple waves
        predefined_paths.append(generate_wave_path(num_points, amplitude, wavelength, center, num_waves))
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
                # Only check points between the start and end gaps
                if start_gap <= i < len(predefined_paths[current_path]) - end_gap:
                    if not touched_points[current_path][i] and np.linalg.norm(np.array([finger_x, finger_y]) - np.array(path_point)) < 20:
                        touched_points[current_path][i] = True

    # Draw the current predefined path, updating colors based on touched status
    draw_path(image, predefined_paths[current_path], touched_points[current_path], start_gap, end_gap)

    # Check if the last `num_ending_circles` circles have been touched
    if all(touched_points[current_path][-num_ending_circles - end_gap:-end_gap]):
        # Calculate the percentage of circles touched
        total_circles = len(touched_points[current_path]) - start_gap - end_gap
        touched_circles = sum(touched_points[current_path][start_gap:len(touched_points[current_path]) - end_gap])
        percentage_touched = (touched_circles / total_circles) * 100

        # Show the percentage of touched circles and exit
        print(f"Percentage of circles touched: {percentage_touched:.2f}%")
        break

    cv2.imshow("Air Drawing - Multiple Waves", image)

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
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))