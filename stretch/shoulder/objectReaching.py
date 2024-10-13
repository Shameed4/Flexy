import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Initialize variables
object_position = None
object_radius = 30
object_touched = False

# Timing variables
object_spawn_time = None
object_display_duration = 5  # seconds

# Counters for touch events
total_spawns = 0
successful_touches = 0

cap = cv2.VideoCapture(1)

# Load the banner image (Adjust the path to your banner image)
banner_image = cv2.imread('3.png')  # Adjust the path to your banner image

# Function to add the banner from an existing image with aspect ratio preservation
def add_banner_from_image(image, banner_image):
    frame_width = image.shape[1]
    banner_aspect_ratio = banner_image.shape[1] / banner_image.shape[0]
    new_banner_width = frame_width
    new_banner_height = int(frame_width / banner_aspect_ratio)

    banner_resized = cv2.resize(banner_image, (new_banner_width, new_banner_height))

    expected_banner_height = 150  # Adjust this for the banner height
    if banner_resized.shape[0] < expected_banner_height:
        padding = np.zeros((expected_banner_height - banner_resized.shape[0], new_banner_width, 3), dtype=np.uint8)
        banner_resized = np.vstack((banner_resized, padding))

    # Concatenate the banner with the image vertically
    full_image = np.vstack((banner_resized, image))
    return full_image

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Mirror image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = frame.shape

    results = hands.process(frame_rgb)

    index_finger_tip = None

    # Spawn a new object if there isn't one or time has expired
    current_time = time.time()
    if object_position is None or (current_time - object_spawn_time) > object_display_duration or object_touched:
        object_x = random.randint(object_radius, width - object_radius)
        object_y = random.randint(object_radius, height - object_radius)
        object_position = [object_x, object_y]
        object_spawn_time = current_time
        object_touched = False
        total_spawns += 1  # Increment total spawns

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

        # Check if the index finger tip is touching the object
        distance = np.linalg.norm(np.array(index_finger_tip) - np.array(object_position))
        if distance < object_radius + 10:  # 10 pixels buffer
            object_touched = True
            successful_touches += 1  # Increment successful touches
            cv2.putText(frame, "Touched!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw the virtual object
    if not object_touched:
        cv2.circle(frame, tuple(object_position), object_radius, (0, 255, 255), -1)  # Yellow circle

    # Display success rate
    if total_spawns > 0:
        success_rate = (successful_touches / total_spawns) * 100
        cv2.putText(frame, f"Success Rate: {success_rate:.2f}%", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Add the banner before displaying the frame
    frame_with_banner = add_banner_from_image(frame, banner_image)

    cv2.imshow("Virtual Object Reaching", frame_with_banner)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
print("100.00")