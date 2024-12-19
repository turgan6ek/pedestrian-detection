import os
import json
from pedestrian_detection import PedestrianDetection
from rl_agent import RLAgent
import matplotlib.pyplot as plt

# Initialize components
detector = PedestrianDetection()
agent = RLAgent()

# Directory paths
input_dir = "./data/videos/"
output_dir_rl = "./runs/detect/rl/"
pseudo_ground_truth_dir = "./runs/ground_truth/"

# Ensure directories exist
os.makedirs(output_dir_rl, exist_ok=True)
os.makedirs(pseudo_ground_truth_dir, exist_ok=True)

# Lists for logging parameter changes
confidence_values = []
iou_values = []

# Process all videos in the input directory
for file_name in os.listdir(input_dir):
    if file_name.endswith((".mp4", ".avi", ".mkv")):
        video_path = os.path.join(input_dir, file_name)
        print(f"Processing video: {video_path}")

        # Generate pseudo ground truth
        pseudo_ground_truth_file = os.path.join(pseudo_ground_truth_dir, f"{file_name}_ground_truth.json")
        ground_truth = detector.generate_pseudo_ground_truth(video_path, pseudo_ground_truth_file)

        # Load pseudo ground truth
        with open(pseudo_ground_truth_file, "r") as f:
            ground_truth = json.load(f)

        # Validate ground truth format
        ground_truth = [gt for gt in ground_truth if len(gt) == 4]

        # RL loop
        state = (5, 5)  # Initial state
        for frame_idx in range(10):  # Simulated loop for 10 frames
            # RL agent decides YOLO parameters
            action = agent.choose_action(state)
            conf, iou = action[0] / 10, action[1] / 10
            confidence_values.append(conf)
            iou_values.append(iou)

            # Perform YOLO detection
            results = detector.process_video(video_path, conf=conf, iou=iou, output_dir=output_dir_rl)

            # Extract RL detections
            detections = [box.xyxy.tolist() for box in results[0].boxes if len(box.xyxy.tolist()) == 4]

            # Calculate reward
            reward = agent.reward_function(detections, ground_truth)
            next_state = action

            # Update RL agent
            agent.update(state, action, reward, next_state)
            state = next_state  # Transition to the next state

# Plot parameter changes
plt.figure(figsize=(10, 5))
plt.plot(confidence_values, label="Confidence")
plt.plot(iou_values, label="IoU Threshold")
plt.xlabel("Frame")
plt.ylabel("Parameter Value")
plt.title("YOLO Parameters Over Frames")
plt.legend()
plt.show()

# Plot Q-table
agent.plot_q_table()
