import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Define the path (a simple curve in this example)
def generate_path(num_points, width, height):
    path = []
    for t in np.linspace(0, np.pi, num_points):
        x = int(width * (0.1 + 0.8 * t / np.pi))
        y = int(height * (0.5 + 0.4 * np.sin(t)))
        path.append((x, y))
    return path

# Initialize variables
num_points = 100
cap = cv2.VideoCapture(0)
success, frame = cap.read()
height, width, _ = frame.shape

path = generate_path(num_points, width, height)
touched_points = [False] * num_points
current_point_index = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Mirror image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = frame.shape

    results = hands.process(frame_rgb)

    index_finger_tip = None

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark

        # Get coordinates of index finger tip
        index_finger_tip = (
            int(lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width),
            int(lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
        )

        # Draw index finger tip
        cv2.circle(frame, index_finger_tip, 10, (255, 0, 0), -1)

        # Check if index finger tip is near the current point in the path
        target_point = path[current_point_index]
        distance = np.linalg.norm(np.array(index_finger_tip) - np.array(target_point))
        if distance < 20:
            touched_points[current_point_index] = True
            current_point_index += 1
            if current_point_index >= num_points:
                cv2.putText(frame, "Completed!", (width//2 - 100, height//2), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Arm Movement Path Following", frame)
                cv2.waitKey(2000)
                break

    # Draw the path
    for i, point in enumerate(path):
        color = (0, 0, 255) if not touched_points[i] else (0, 255, 0)
        cv2.circle(frame, point, 5, color, -1)

    cv2.imshow("Arm Movement Path Following", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
