import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Predefined path (example points)
# Predefined path: circle points
num_points = 8  # Number of points in the circle
radius = 200  # Radius of the circle
center = (320, 240)  # Center of the circle (assuming 640x480 resolution)

# Generate circle points
predefined_path = [
    (int(center[0] + radius * np.cos(2 * np.pi * i / num_points)),
     int(center[1] + radius * np.sin(2 * np.pi * i / num_points)))
    for i in range(num_points)
]

# Track touched status for each point in the predefined path
touched_points = [False] * num_points

# Function to draw the predefined path
def draw_path(image, path, touched_points):
    for i, point in enumerate(path):
        color = (0, 0, 255) if touched_points[i] else (0, 255, 0)
        cv2.circle(image, point, 10, color, -1)

cap = cv2.VideoCapture(1)

while True:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)  # Flip the image for a mirror effect
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # If hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the position of the index finger tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            finger_x = int(index_finger_tip.x * image.shape[1])
            finger_y = int(index_finger_tip.y * image.shape[0])

            # Draw the finger position
            cv2.circle(image, (finger_x, finger_y), 10, (0, 0, 255), -1)

            # Check if finger is near predefined path (simple distance check)
            for i, path_point in enumerate(predefined_path):
                if np.linalg.norm(np.array([finger_x, finger_y]) - np.array(path_point)) < 20:
                    touched_points[i] = True

    # Draw the predefined path, updating colors based on touched status
    draw_path(image, predefined_path, touched_points)

    cv2.imshow("Air Drawing", image)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()