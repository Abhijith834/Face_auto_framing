import cv2
import numpy as np

# Load the model for face detection
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

def detect_face(frame, net):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            return (startX, startY, endX - startX, endY - startY)  # Return as x, y, w, h
    return None

# Initialize tracker
def initialize_tracker(frame, bbox):
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)
    return tracker

# Track and keep the face centered
def track_and_center_face(frame, tracker):
    success, bbox = tracker.update(frame)
    if success:
        x, y, w, h = [int(v) for v in bbox]
        # Optional: Draw the bounding box
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2, 1)
        frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
        face_center = (x + w // 2, y + h // 2)
        dx, dy = frame_center[0] - face_center[0], frame_center[1] - face_center[1]

        # Simple method to keep face centered (may adjust for smoother transitions)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

        return True, shifted_frame
    return False, frame

# Initialize webcam
cap = cv2.VideoCapture(0)

tracker = None
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if tracker is None:
        bbox = detect_face(frame, net)
        if bbox is not None:
            tracker = initialize_tracker(frame, bbox)
    else:
        success, frame = track_and_center_face(frame, tracker)
        if not success:
            tracker = None  # Reset tracker if tracking fails

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
