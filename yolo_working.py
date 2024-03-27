from ultralytics import YOLO

# Create a new instance of the YOLO model
model = YOLO('yolov8x')

# Use the YOLO model to predict the objects in the image
result = model.predict('TpImages/fb_img.jpeg', save=True)

# Print the result of the object detection
print(result)

# Print the labels and confidence scores of the detected objects
print("Labels and Confidence Scores:")
for obj in result[0].objects:
    print(f"Label: {obj.class_name}, Confidence: {obj.confidence}")
