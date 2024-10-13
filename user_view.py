import cv2
import numpy as np
import mediapipe as mp
import math
import json
import os
import argparse
from natsort import natsorted

import argparse

# Set up argparse
parser = argparse.ArgumentParser(description="Process pose arguments.")

poses_dir = "./poses"

# customizable parts
exercise = "arm_circles"
sensitivity = 0.9

def update_from_pose(path):
    with open(f"{path}.json", "r") as f:
        data = json.load(f)
    global guide_image, landmarks_of_interest, connections, guide_pose
    guide_image = cv2.imread(f"{path}.png")
    landmarks_of_interest = data["landmarks"]
    connections = data["connections"]
    guide_pose = data["coordinates"]

print_feedback = True
show_guide = True

# Process arguments
extra_directory = ""
working_directory = "."

# Initialize dictionaries to store saved poses
guide_pose = {}

poses_to_complete = natsorted([f"{poses_dir}/{exercise}/{item.split(".")[0]}" for item in os.listdir(f"{poses_dir}/{exercise}") if item.endswith(".json")])
current_exercise = 0
update_from_pose(poses_to_complete[0])

if not os.path.exists(working_directory):
    os.makedirs(working_directory)
    
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

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
        return 100
    if end not in current_pose:
        feedback.append(f"{end} not detected")
        return 100
    
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
    if print_feedback and feedback != []:
        print(feedback)
    # Combine scores (you may want to adjust the weights)
    total_score = sum(direction_scores)
    return total_score

def is_matching_pose(similarity):
    return similarity < sensitivity

# Start video capture
cap = cv2.VideoCapture(0)

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

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

        # Draw connections
        for (start, end) in connections:
            if start in current_pose and end in current_pose:
                start_coord = current_pose[start]
                end_coord = current_pose[end]
                start_point = (int(start_coord['x'] * frame.shape[1]), int(start_coord['y'] * frame.shape[0]))
                end_point = (int(end_coord['x'] * frame.shape[1]), int(end_coord['y'] * frame.shape[0]))
                cv2.line(frame, start_point, end_point, (255, 0, 0), 2)  # Blue line
                
    # Draw guide connections
    if show_guide and guide_image is not None:
        # Define the transparency level for the overlay (0.0 = fully transparent, 1.0 = fully opaque)
        alpha = 0.7

        # Blend the images using the alpha value
        frame = cv2.addWeighted(guide_image, alpha, frame, 1 - alpha, 0)
        # Draw landmarks and labels
        for name, coord in guide_pose.items():
            x, y = int(coord['x'] * frame.shape[1]), int(coord['y'] * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)  # Draw the landmark
        for (start, end) in connections:
            if start in current_pose and end in current_pose:
                start_coord = guide_pose[start]
                end_coord = guide_pose[end]
                start_point = (int(start_coord['x'] * frame.shape[1]), int(start_coord['y'] * frame.shape[0]))
                end_point = (int(end_coord['x'] * frame.shape[1]), int(end_coord['y'] * frame.shape[0]))
                cv2.line(frame, start_point, end_point, (255, 0, 155), 2)  # Purple line

    # Check for the 's' key to start the timer for the first saved pose
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('f'):
        print_feedback = not print_feedback
        if print_feedback:
            print("Printing feedback")
        else:
            print("Not printing feedback")
    
    if key == ord('g'):
        show_guide = not show_guide
        if show_guide:
            print("Displaying guide")
        else:
            print("Hiding guide")

    # If saved poses exist, calculate and display similarity
    similarity_score_s = calculate_total_similarity(current_pose, guide_pose)
    cv2.putText(frame, f"Similarity Score S: {similarity_score_s:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if exercise is not None and is_matching_pose(similarity_score_s):
        current_exercise += 1
        if current_exercise == len(poses_to_complete):
            print("Success!")
            break   
        update_from_pose(poses_to_complete[current_exercise])

    # Show the frame
    cv2.imshow('Push-Up Counter', frame)

# Release resources
cap.release()
cv2.destroyAllWindows()
