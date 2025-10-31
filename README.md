Name:Hemavardhan raju
ID:CA/SE1/18250
Domain:Artificial intelligence
Duration:10th OCT 2025 to 10th NOV 2025
# CodeAlpha
import cv2
import torch
import numpy as np

# Install a SORT implementation: pip install sort
from sort import Sort  

# --- Initialize video capture ---
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or 'video.mp4' for file

# --- Load YOLOv5 model ---
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4  # confidence threshold (0-1)

# --- Initialize SORT tracker ---
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Object Detection ---
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, conf, class

    # --- Prepare detections for SORT (bbox + confidence) ---
    dets_for_sort = detections[:, :5]  # x1, y1, x2, y2, conf

    # --- Object Tracking ---
    tracked_objects = tracker.update(dets_for_sort)

    # --- Draw results ---
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = obj
        x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put track ID
        cv2.putText(frame, f'ID: {track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Show output
    cv2.imshow("Object Detection + Tracking", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
<img width="490" height="127" alt="Screenshot 2025-10-31 221347" src="https://github.com/user-attachments/assets/74d5cc5c-1051-44ff-a113-4b1e6e9b9aff" />
<img width="537" height="378" alt="Screenshot 2025-10-31 221358" src="https://github.com/user-attachments/assets/a60aa497-4cb6-44cb-8c9c-8bb021f8aa82" />

