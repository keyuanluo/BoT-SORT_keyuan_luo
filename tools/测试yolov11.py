# from ultralytics import settings
#
# # Update a setting
# settings.update({"runs_dir": "/path/to/runs"})
#
# # Update multiple settings
# settings.update({"runs_dir": "/path/to/runs", "tensorboard": False})
#
# # Reset settings to default values
# settings.reset()

from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("yolo11n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo11n.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="coco8.yaml", epochs=3)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model("https://ultralytics.com/images/bus.jpg")

# Export the model to ONNX format
success = model.export(format="onnx")