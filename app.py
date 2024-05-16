import csv

from flask import Flask, render_template

app = Flask(__name__)

# Import necessary libraries for YOLO detection
from ultralytics import YOLO
import cv2
import cvzone
import math
import time

# Initialize YOLO model
model = YOLO("../Yolo-Weights/best.pt")

# Define class names
classNames = ["anger", "disgust", "fear", "happy", "neutral", "sad", "neutral"]

# Route to handle emotion detection
@app.route('/detect_emotion')
def detect_emotion():
    cap = cv2.VideoCapture(0)  # For Webcam
    cap.set(3, 1280)
    cap.set(4, 720)

    prev_frame_time = 0
    new_frame_time = 0

    while True:
        new_frame_time = time.time()
        success, img = cap.read()
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        print(fps)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Route to render the index.html template
@app.route('/')
def home():
    # Read the CSV file
    with open('spotify.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)  # Convert the reader to a list of dictionaries

    return render_template('index.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)