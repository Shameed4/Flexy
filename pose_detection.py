import cv2
import numpy as np
import mediapipe as mp
import math
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Define landmarks of interest and their connections
landmarks_of_interest = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}

connections = [
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
]

# Initialize dictionaries to store saved poses
saved_pose_s = {}
saved_pose_r = {}
save_timer_s = None  # Timer variable for 's' key countdown
save_timer_r = None  # Timer variable for 'r' key countdown
countdown_duration = 2  # 2-second delay for pose save

# Initialize counters
push_up_count = 0
r_previous_greater = False  # Track if previous R score was greater than S score
down_to_resting_count = 0  # Track transitions from down to resting
current_state = "Neither"  # Track current state of the position

# Function to calculate angle between two vectors
def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (norm_v1 * norm_v2)
    angle = math.acos(np.clip(cos_angle, -1.0, 1.0))  # Clamp to avoid floating point errors
    return angle

# Function to calculate position similarity
def calculate_position_similarity(current_pose, saved_pose):
    position_score = 0
    for name in landmarks_of_interest.keys():
        if name in current_pose and name in saved_pose:
            curr_coord = current_pose[name]
            saved_coord = saved_pose[name]
            dist = np.linalg.norm(np.array([curr_coord['x'], curr_coord['y']]) -
                                  np.array([saved_coord['x'], saved_coord['y']]))
            position_score += dist
    return position_score

# Function to calculate direction similarity
def calculate_direction_similarity(current_pose, saved_pose):
    total_angle_difference = 0
    for (start, end) in connections:
        if start in current_pose and start in saved_pose and end in current_pose and end in saved_pose:
            curr_start = current_pose[start]
            curr_end = current_pose[end]
            saved_start = saved_pose[start]
            saved_end = saved_pose[end]

            curr_vector = np.array([curr_end['x'] - curr_start['x'], curr_end['y'] - curr_start['y']])
            saved_vector = np.array([saved_end['x'] - saved_start['x'], saved_end['y'] - saved_start['y']])
            angle_diff = calculate_angle(curr_vector, saved_vector)
            total_angle_difference += angle_diff

    return total_angle_difference

# Function to calculate total similarity
def calculate_total_similarity(current_pose, saved_pose):
    position_score = calculate_position_similarity(current_pose, saved_pose)
    direction_score = calculate_direction_similarity(current_pose, saved_pose)

    # Combine scores (you may want to adjust the weights)
    total_score = position_score + direction_score
    return total_score

# Start video capture
cap = cv2.VideoCapture(0)

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the image and get pose landmarks
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    current_pose = {}

    if results.pose_landmarks:
        for name, index in landmarks_of_interest.items():
            landmark = results.pose_landmarks.landmark[index]
            current_pose[name] = {'x': landmark.x, 'y': landmark.y}

        # Draw landmarks and labels
        for name, coord in current_pose.items():
            x, y = int(coord['x'] * frame.shape[1]), int(coord['y'] * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw the landmark
            cv2.putText(frame, name, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # Draw label

        # Draw connections
        for (start, end) in connections:
            if start in current_pose and end in current_pose:
                start_coord = current_pose[start]
                end_coord = current_pose[end]
                start_point = (int(start_coord['x'] * frame.shape[1]), int(start_coord['y'] * frame.shape[0]))
                end_point = (int(end_coord['x'] * frame.shape[1]), int(end_coord['y'] * frame.shape[0]))
                cv2.line(frame, start_point, end_point, (255, 0, 0), 2)  # Blue line

    # Check for the 's' key to start the timer for the first saved pose
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        save_timer_s = time.time()  # Set the start time for 's' countdown
        print("Get ready! Saving pose in 2 seconds...")

    # Check for the 'r' key to start the timer for the second saved pose
    if key == ord('r'):
        save_timer_r = time.time()  # Set the start time for 'r' countdown
        print("Get ready! Saving pose in 2 seconds...")

    # If timer for 's' key started, check if countdown duration has passed
    if save_timer_s and (time.time() - save_timer_s >= countdown_duration):
        # Capture and save the pose
        saved_pose_s = current_pose.copy()
        print("Pose 's' saved.")
        cv2.imwrite("saved_pose_s.png", frame)  # Save the current frame as an image
        save_timer_s = None  # Reset timer after saving

    # If timer for 'r' key started, check if countdown duration has passed
    if save_timer_r and (time.time() - save_timer_r >= countdown_duration):
        # Capture and save the pose
        saved_pose_r = current_pose.copy()
        print("Pose 'r' saved.")
        cv2.imwrite("saved_pose_r.png", frame)  # Save the current frame as an image
        save_timer_r = None  # Reset timer after saving

    # If saved poses exist, calculate and display similarity
    similarity_score_s = None
    similarity_score_r = None
    if saved_pose_s:
        similarity_score_s = calculate_total_similarity(current_pose, saved_pose_s)
        cv2.putText(frame, f"Similarity Score S: {similarity_score_s:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if saved_pose_r:
        similarity_score_r = calculate_total_similarity(current_pose, saved_pose_r)
        cv2.putText(frame, f"Similarity Score R: {similarity_score_r:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Determine current position
    is_resting = similarity_score_r is not None and similarity_score_r < 2 and (similarity_score_s is not None and similarity_score_s > 5)
    is_down = similarity_score_r is not None and similarity_score_r > 5 and (similarity_score_s is not None and similarity_score_s < 2)

    # Check for transitions and update counts
    if is_down:
        current_state = "Down"
        if r_previous_greater:
            down_to_resting_count += 1
            print("Transition from Down to Resting counted! Total transitions:", down_to_resting_count)
            r_previous_greater = False  # Reset the flag after counting
    elif is_resting:
        current_state = "Resting"
        r_previous_greater = True  # Set the flag if R is greater
    else:
        current_state = "Neither"

    # Display the current state and counts
    cv2.putText(frame, f"Current State: {current_state}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Push-up Count: {push_up_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Down to Resting Count: {down_to_resting_count}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Check for 'X' key to reset the counter
    if key == ord('x'):
        down_to_resting_count = 0
        print("Counter reset!")

    # Show the frame
    cv2.imshow('Push-Up Counter', frame)

# Release resources
cap.release()
cv2.destroyAllWindows()
