import cv2
import numpy as np
import mediapipe as mp
import math
import time
import json
import sys
import os

print_feedback = False
loop_saving = False
extra_directory = ""
downloaded_count = 0
if len(sys.argv) > 1:
    extra_directory = sys.argv[1]
working_directory = f"./poses/{extra_directory}"
if not os.path.exists(working_directory):
    os.makedirs(working_directory)
    
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

# Function to calculate direction similarity
def calculate_direction_similarity(start, end, current_pose, saved_pose, feedback):
    if start not in current_pose:
        feedback.append(f"{start} not detected")
        return 10
    if end not in current_pose:
        feedback.append(f"{end} not detected")
        return 10
    
    curr_start = current_pose[start]
    curr_end = current_pose[end]
    saved_start = saved_pose[start] 
    saved_end = saved_pose[end]

    curr_vector = np.array([curr_end['x'] - curr_start['x'], curr_end['y'] - curr_start['y']])
    saved_vector = np.array([saved_end['x'] - saved_start['x'], saved_end['y'] - saved_start['y']])
    angle_diff = calculate_angle(curr_vector, saved_vector)

    if angle_diff > 0.5:
        feedback.append(f"Angle difference is high between {start} and {end}")
        
    
    return angle_diff

# Function to calculate total similarity
def calculate_total_similarity(current_pose, saved_pose):
    feedback = []
    direction_scores = [calculate_direction_similarity(start, end, current_pose, saved_pose, feedback) for start, end in connections]
    if print_feedback:
        print(feedback)
    # Combine scores (you may want to adjust the weights)
    total_score = sum(direction_scores)
    return total_score

def download_saved_pose():
    print("Downloading saved pose")
    global downloaded_count
    cv2.imwrite(f"{working_directory}/pose_{downloaded_count}.png", saved_image_s)  # Save the current frame as an image
    with open(f"{working_directory}/pose_{downloaded_count}.json", "w") as f:
        json.dump({"coordinates": saved_pose_s, "connections": connections}, f, indent=2, separators=(", ", ": "))
    downloaded_count += 1

# Start video capture
cap = cv2.VideoCapture(0)

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the image and get pose landmarks
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # If timer for 's' key started, check if countdown duration has passed
    if save_timer_s and (time.time() - save_timer_s >= countdown_duration):
        # Capture and save the pose
        saved_pose_s = current_pose.copy()
        saved_image_s = frame
        print("Pose 's' saved.")
        if loop_saving:
            save_timer_s = time.time()
            download_saved_pose()
        else:        
            save_timer_s = None  # Reset timer after saving

    # If timer for 'r' key started, check if countdown duration has passed
    if save_timer_r and (time.time() - save_timer_r >= countdown_duration):
        # Capture and save the pose
        saved_pose_r = current_pose.copy()
        saved_image_r = frame
        print("Pose 'r' saved.")
        save_timer_r = None  # Reset timer after saving
    
    
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
        print(f"Get ready! Saving pose in {countdown_duration} seconds...")

    # Check for the 'r' key to start the timer for the second saved pose
    if key == ord('r'):
        save_timer_r = time.time()  # Set the start time for 'r' countdown
        print("Get ready! Saving pose in 2 seconds...")
    
    if key == ord('d'):
        download_saved_pose()
    
    if key == ord('l'):
        loop_saving = not loop_saving
        if loop_saving:
            print("Looping saving and downloading")
        else:
            print("Disabled looping saving and downloading")
    
    if key == ord('f'):
        print_feedback = not print_feedback
        if print_feedback:
            print("Printing feedback")
        else:
            print("Not printing feedback")

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
