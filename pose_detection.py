import cv2
import numpy as np
import mediapipe as mp
import math
import time
import json
import os

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
saved_poses = {}
save_timers = {}
countdown_duration = 2  # 2-second delay for pose save
reps_completed = 0  # Count of completed reps
sequence_index = 0  # Index in the expected sequence
expected_sequence = ['1', '2', '3']  # Default expected sequence (can be extended up to '9')

# Variables for pose holding and points
pose_hold_start_time = None
is_in_pose = False
pose_hold_duration = 0
pose_hold_threshold = 5  # seconds
points = 0
feedback_threshold = 0.5  # Threshold for generating feedback

# Load saved poses from JSON file if it exists
saved_poses_file = 'saved_poses.json'
if os.path.exists(saved_poses_file):
    with open(saved_poses_file, 'r') as f:
        saved_poses = json.load(f)
    print(f"Loaded saved poses from {saved_poses_file}")

# Function to calculate angle between two vectors
def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (norm_v1 * norm_v2)
    angle = math.acos(np.clip(cos_angle, -1.0, 1.0))  # Clamp to avoid floating point errors
    return angle

# Function to calculate joint angle at point b formed by points a, b, c
def calculate_joint_angle(a, b, c):
    ba = np.array([a['x'] - b['x'], a['y'] - b['y']])
    bc = np.array([c['x'] - b['x'], c['y'] - b['y']])
    angle = calculate_angle(ba, bc)
    return angle

# Function to calculate position similarity
def calculate_position_similarity(current_pose, saved_pose):
    position_score = 0
    count = 0
    for name in landmarks_of_interest.keys():
        if name in current_pose and name in saved_pose:
            curr_coord = current_pose[name]
            saved_coord = saved_pose[name]
            dist = np.linalg.norm(np.array([curr_coord['x'], curr_coord['y']]) -
                                  np.array([saved_coord['x'], saved_coord['y']]))
            position_score += dist
            count += 1
    if count > 0:
        position_score /= count  # Average distance per landmark
    return position_score

# Function to calculate direction similarity
def calculate_direction_similarity(current_pose, saved_pose):
    total_angle_difference = 0
    count = 0
    for (start, end) in connections:
        if start in current_pose and start in saved_pose and end in current_pose and end in saved_pose:
            curr_start = current_pose[start]
            curr_end = current_pose[end]
            saved_start = saved_pose[start]
            saved_end = saved_pose[end]

            curr_vector = np.array([curr_end['x'] - curr_start['x'], curr_end['y'] - curr_start['y']])
            saved_vector = np.array([saved_end['x'] - saved_start['x'], saved_end['y'] - saved_start['y']])
            angle_diff = calculate_angle(curr_vector, saved_vector) / math.pi  # Normalize angle difference
            total_angle_difference += angle_diff
            count += 1

    if count > 0:
        total_angle_difference /= count  # Average normalized angle difference per connection
    return total_angle_difference

# Function to calculate total similarity
def calculate_total_similarity(current_pose, saved_pose):
    # Compute a scaling factor, e.g., the distance between shoulders in the saved pose
    if 'left_shoulder' in saved_pose and 'right_shoulder' in saved_pose:
        saved_scale = np.linalg.norm(np.array([saved_pose['left_shoulder']['x'], saved_pose['left_shoulder']['y']]) -
                                     np.array([saved_pose['right_shoulder']['x'], saved_pose['right_shoulder']['y']]))
    else:
        saved_scale = 1  # Default to 1 if not available

    # Similarly for current pose
    if 'left_shoulder' in current_pose and 'right_shoulder' in current_pose:
        current_scale = np.linalg.norm(np.array([current_pose['left_shoulder']['x'], current_pose['left_shoulder']['y']]) -
                                       np.array([current_pose['right_shoulder']['x'], current_pose['right_shoulder']['y']]))
    else:
        current_scale = 1  # Default to 1 if not available

    scale = (saved_scale + current_scale) / 2  # Average scale

    position_score = calculate_position_similarity(current_pose, saved_pose) / scale
    direction_score = calculate_direction_similarity(current_pose, saved_pose)

    # Combine scores with weights (adjust weights as needed)
    total_score = 0.4 * position_score + 0.6 * direction_score
    return total_score

# Function to generate feedback based on pose differences
def generate_feedback(current_pose, saved_pose):
    feedback = []
    position_threshold = 0.05  # Adjust as needed
    angle_threshold = 0.15  # Radians (~8.6 degrees)

    # Compare y position of shoulders
    for side in ['left', 'right']:
        shoulder = f'{side}_shoulder'
        if shoulder in current_pose and shoulder in saved_pose:
            curr_y = current_pose[shoulder]['y']
            saved_y = saved_pose[shoulder]['y']
            diff_y = curr_y - saved_y  # Positive if current is lower in the image
            if abs(diff_y) > position_threshold:
                if diff_y > 0:
                    feedback.append(f'Raise your {side} shoulder')
                else:
                    feedback.append(f'Lower your {side} shoulder')

    # Compare angles at elbows
    for side in ['left', 'right']:
        shoulder = f'{side}_shoulder'
        elbow = f'{side}_elbow'
        wrist = f'{side}_wrist'
        if shoulder in current_pose and elbow in current_pose and wrist in current_pose and \
           shoulder in saved_pose and elbow in saved_pose and wrist in saved_pose:
            # Calculate current angle at elbow
            curr_angle = calculate_joint_angle(current_pose[shoulder], current_pose[elbow], current_pose[wrist])
            # Calculate saved angle at elbow
            saved_angle = calculate_joint_angle(saved_pose[shoulder], saved_pose[elbow], saved_pose[wrist])
            diff_angle = curr_angle - saved_angle
            if abs(diff_angle) > angle_threshold:
                if diff_angle > 0:
                    feedback.append(f'Straighten your {side} elbow')
                else:
                    feedback.append(f'Bend your {side} elbow more')

    # Additional feedback can be added for other joints or landmarks

    return feedback

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
            current_pose[name] = {'x': float(landmark.x), 'y': float(landmark.y)}

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

    # Check for the '1'-'9' keys to start the timer for saved poses
    key = cv2.waitKey(1) & 0xFF
    if ord('1') <= key <= ord('9'):
        pose_index = chr(key)  # Convert ASCII to character for indexing
        save_timers[pose_index] = time.time()  # Set the start time for the pose countdown
        print(f"Get ready! Saving pose {pose_index} in {countdown_duration} seconds...")

    # If timer for a key started, check if countdown duration has passed
    for pose_index in list(save_timers.keys()):
        if save_timers[pose_index] and (time.time() - save_timers[pose_index] >= countdown_duration):
            # Capture and save the pose
            saved_poses[pose_index] = current_pose.copy()
            print(f"Pose {pose_index} saved.")
            save_timers[pose_index] = None  # Reset timer after saving

            # Save saved_poses to JSON file
            with open(saved_poses_file, 'w') as f:
                json.dump(saved_poses, f)
            print(f"Saved poses to {saved_poses_file}")

    # If saved poses exist, calculate and display similarity
    similarity_scores = {}
    for pose_index, saved_pose in saved_poses.items():
        similarity_scores[pose_index] = calculate_total_similarity(current_pose, saved_pose)

    # Determine current pose based on similarity scores
    current_pose_index = None
    if similarity_scores:
        # Find the pose with the minimum score
        min_pose_index = min(similarity_scores, key=similarity_scores.get)
        min_score = similarity_scores[min_pose_index]

        # Set a threshold for considering a pose match
        threshold = 0.5  # Adjust this threshold based on testing
        if min_score < threshold:
            current_pose_index = min_pose_index
            is_pose_correct = True
        else:
            is_pose_correct = False
    else:
        is_pose_correct = False

    # Handle pose holding and point awarding
    if is_pose_correct:
        if not is_in_pose:
            is_in_pose = True
            pose_hold_start_time = time.time()
            pose_hold_duration = 0
        else:
            pose_hold_duration = time.time() - pose_hold_start_time
            if pose_hold_duration >= pose_hold_threshold:
                points += 1
                print(f"Pose held for {pose_hold_threshold} seconds. Points awarded: {points}")
                # Reset the timer to continue awarding points every pose_hold_threshold seconds
                pose_hold_start_time = time.time()
    else:
        if is_in_pose:
            is_in_pose = False
            pose_hold_start_time = None
            pose_hold_duration = 0

    # Generate feedback if not in correct pose
    feedback_messages = []
    if not is_pose_correct and min_pose_index in saved_poses:
        feedback_messages = generate_feedback(current_pose, saved_poses[min_pose_index])

    # Display similarity scores and current pose
    y_offset = 50
    for index, score in similarity_scores.items():
        text = f'Pose {index}: {score:.2f}'
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20

    if is_pose_correct:
        expected_pose_text = f'Pose Matched: {current_pose_index}'
    else:
        expected_pose_text = 'Pose Not Matched'
    cv2.putText(frame, expected_pose_text, (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # Display feedback messages
    for i, msg in enumerate(feedback_messages):
        cv2.putText(frame, msg, (10, y_offset + 50 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display pose hold duration and points
    if is_in_pose:
        cv2.putText(frame, f'Hold Time: {pose_hold_duration:.1f}s', (10, y_offset + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f'Points: {points}', (10, y_offset + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Show the frame with drawn landmarks and pose data
    cv2.imshow('Exercise Assistant', frame)

    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
