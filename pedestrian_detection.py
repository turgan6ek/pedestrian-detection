import os
import json
from ultralytics import YOLO

class PedestrianDetection:
    def __init__(self, model_path="yolov8n.pt"):
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        print("Model loaded successfully!")

    def process_video(self, video_path, conf=0.5, iou=0.5, output_dir="../runs/detect/"):
        """
        Processes a single video file using YOLO with specified confidence and IoU thresholds.
        """
        print(f"Processing video: {video_path} with Confidence={conf}, IoU={iou}")
        results = self.model.predict(source=video_path, conf=conf, iou=iou, save=True, project=output_dir)
        print(f"Finished processing: {video_path}")
        print(f"Results saved in: {output_dir}")
        return results

    def generate_pseudo_ground_truth(self, video_path, output_file):
        """
        Generates pseudo ground truth by running YOLO with fixed parameters.
        """
        results = self.model.predict(source=video_path, conf=0.5, iou=0.5, save=False)
        ground_truth = [box.xyxy.tolist() for box in results[0].boxes]

        with open(output_file, "w") as f:
            json.dump(ground_truth, f)

        print(f"Pseudo ground truth saved to {output_file}")
        return ground_truth
