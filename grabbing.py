import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Initialize variables
object_position = [100, 100]  # Starting position of the object
object_radius = 30
object_grabbed = False

target_position = [400, 300]  # Position of the target area
target_radius = 40

grab_threshold = 30  # Threshold distance to consider the object grabbed
release_threshold = 50  # Threshold to detect release (between thumb and index)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # Mirror image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = frame.shape

    results = hands.process(frame_rgb)

    index_tip = None
    thumb_tip = None

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark

        # Get coordinates of index finger tip and thumb tip
        index_tip = (
            int(lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width),
            int(lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
        )
        thumb_tip = (
            int(lm[mp_hands.HandLandmark.THUMB_TIP].x * width),
            int(lm[mp_hands.HandLandmark.THUMB_TIP].y * height)
        )

        # Calculate distance between thumb and index finger tips
        pinch_distance = np.linalg.norm(np.array(index_tip) - np.array(thumb_tip))

        # Check if the hand is in grabbing position
        if pinch_distance < grab_threshold:
            # Check if the hand is over the object or already grabbed
            hand_position = (
                int((index_tip[0] + thumb_tip[0]) / 2),
                int((index_tip[1] + thumb_tip[1]) / 2)
            )
            object_to_hand_distance = np.linalg.norm(np.array(object_position) - np.array(hand_position))

            if object_to_hand_distance < object_radius or object_grabbed:
                object_grabbed = True
                # Update object position to follow the hand
                object_position = list(hand_position)
        else:
            object_grabbed = False

    # Draw the virtual object
    if object_grabbed:
        cv2.circle(frame, tuple(object_position), object_radius, (0, 0, 255), -1)  # Red when grabbed
    else:
        cv2.circle(frame, tuple(object_position), object_radius, (255, 0, 0), -1)  # Blue when not grabbed

    # Draw the target area
    cv2.circle(frame, tuple(target_position), target_radius, (0, 255, 0), 2)  # Green circle

    # Check if object is in the target area
    distance_to_target = np.linalg.norm(np.array(object_position) - np.array(target_position))
    if distance_to_target < (object_radius + target_radius):
        cv2.putText(frame, "Success!", (width//2 - 50, height//2), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        cv2.imshow("Object Transfer Exercise", frame)
        cv2.waitKey(2000)  # Wait for 2 seconds
        break

    # Optionally, draw hand landmarks for debugging
    if results.multi_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Object Transfer Exercise", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
