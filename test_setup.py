from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Run a prediction
results = model.predict(source="../data/videos/video_0001.mp4", save=True)
print("YOLO Setup Test Successful!")
