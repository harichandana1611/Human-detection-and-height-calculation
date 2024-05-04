import cv2
import numpy as np

# Load pre-trained model from disk
net = cv2.dnn.readNetFromCaffe(
    r'C:\Users\haric\Desktop\HVAC\MobileNetSSD_deploy.prototxt.txt',
    r'C:\Users\haric\Desktop\HVAC\MobileNetSSD_deploy.caffemodel')

classNames = {15: 'person'}

cap = cv2.VideoCapture(0)  # Use only one camera feed

if not cap.isOpened():
    print("Error: Camera failed to open.")
    exit()

def calculate_dimension(start_pixel, end_pixel, reference_size_meters, reference_pixels):
    detected_pixels = abs(end_pixel - start_pixel)
    meter_per_pixel = reference_size_meters / reference_pixels
    dimension_meters = meter_per_pixel * detected_pixels
    return dimension_meters

def classify_height(height_cm):
    if 0 <= height_cm <= 151:
        return 1
    elif 152 <= height_cm <= 166:
        return 2
    elif 167 <= height_cm <= 181:
        return 3
    elif 182 <= height_cm <= 198:
        return 4
    else:
        return 5

# Dictionary to track height counts over time
height_counts_history = {1: [], 2: [], 3: [], 4: [], 5: []}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from the camera.")
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    height_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}  # Reset height counts for each frame

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])
        if idx in classNames and confidence > 0.5:  # Check confidence level
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Calculate height in centimeters
            human_height = calculate_dimension(startY, endY, 1.75, 1000)  # Example values, require real calibration

            # Calculate width in centimeters
            human_width = calculate_dimension(startX, endX, 0.5, 500)  # Example values, require real calibration

            # Classify height and update counts
            height_category = classify_height(human_height)
            height_counts[height_category] += 1

            label = f"{classNames[idx]}: {confidence*100:.2f}% Height: {human_height:.2f}m Width: {human_width:.2f}m"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the number of detected persons
    cv2.putText(frame, f"Count: {sum(height_counts.values())}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Update history of height counts
    for category, count in height_counts.items():
        height_counts_history[category].append(count)

    # Display the number of persons in each height category
    height_text = "Height Counts: "
    for category, count in height_counts.items():
        height_text += f"{category}: {count}  "
    cv2.putText(frame, height_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()