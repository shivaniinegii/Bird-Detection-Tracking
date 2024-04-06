
!pip install twilio

import twilio
import cv2
import streamlit as st
from time import time
from ultralytics import YOLO
import torch
from twilio.rest import Client
import numpy as np
from collections import defaultdict
from PIL import Image
from io import BytesIO

# Twilio credentials
account_sid = "ACd358430a5424d4723743b9b8cbf54c33"
auth_token = "fac9f377dd9f3cb9993d046fbc39d887"
twilio_number = "+16825876151"
to_number = "+918010440620"

# Create a Twilio client
client = Client(account_sid, auth_token)

def send_sms(to_number, from_number, object_detected=0):
    print("Sending SMS...")
    message_body = f'ALERT - {object_detected} The chickens are not moving, please check!!'
    message = client.messages.create(
        to=to_number,
        from_=twilio_number,
        body=message_body
    )
    print(message.sid)

class ObjectDetection:
    def __init__(self):
        self.sms_sent = False
        self.model = YOLO(r"D:/project 156/project code and documents/working code/best2.pt")
        self.start_time = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.time_threshold = 60
        self.last_movement_time = time()
        self.frames_without_movement = 0
        self.last_message_time = time()
        self.results = None
        self.track_history = defaultdict(lambda: [])

    def detect_movement(self, centroids):
        current_time = time()

        if current_time - self.last_message_time < self.time_threshold:
            return False

        if not centroids:
            self.last_movement_time = current_time
            self.frames_without_movement = 0
        else:
            elapsed_time = current_time - self.last_movement_time
            if elapsed_time >= self.time_threshold:
                print(f"Sending SMS - Elapsed time without movement: {elapsed_time} seconds")
                self.last_message_time = current_time
                return True
            self.frames_without_movement += 1

            movement_detected = any(self.check_movement_radius(centroid) for centroid in centroids)
            if movement_detected:
                print(f"Sending SMS - Movement detected for centroids : {centroids}")
                self.last_message_time = current_time
                return True

        return False

    def check_movement_radius(self, centroid):
        radius_threshold = 20
        distance_from_center = np.linalg.norm(np.array(centroid) - np.array([640 / 2, 640 / 2]))
        return distance_from_center > radius_threshold

    def predict(self, im0):
        self.results = self.model(im0)
        return self.results

    def display_fps(self, im0):
        fps = 1 / np.round(time() - self.start_time, 2)
        text = f'FPS: {int(fps)}'
        cv2.putText(im0, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    def plot_bboxes(self, results, im0):
        centroids = []
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        confs = results[0].boxes.conf.cpu().tolist()
        names = results[0].names
        conf_threshold = 0.2

        for idx, (box, cls, conf) in enumerate(zip(boxes, clss, confs)):
            if conf >= conf_threshold:
                centroid = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
                centroids.append(centroid)

                label = f"{names[int(cls)]}  {conf:.2f}"

                color = (0, 0, 255)
                cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),  (255, 0, 0), 3)

                text_color = (255, 255, 255)
                bg_color = (255,0,0)
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(im0, (int(box[0]), int(box[1]) - text_size[1] - 5),
                              (int(box[0]) + text_size[0], int(box[1]) - 5), bg_color, -1)
                cv2.putText(im0, label, (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

                track = self.track_history[centroid]
                track.append(centroid)
                if len(track) > 30:
                    track.pop(0)

                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.circle(im0, (track[-1]), 7, (0,0,255), -1)

        return im0, centroids

    def __call__(self, frame):
        self.start_time = time()
        results = self.predict(frame)
        im0, centroids = self.plot_bboxes(results, frame)

        if self.detect_movement(centroids):
            if not self.sms_sent:
                send_sms(to_number, twilio_number, len(centroids))
                self.sms_sent = True
        else:
            self.sms_sent = False

        self.display_fps(im0)

        return im0

# Streamlit App
def main():
    st.title("YOLOv8 Object Detection with SMS Alert")

    capture_source = st.radio("Select Input Source", ["Video File", "Webcam"])

    detector = ObjectDetection()

    if capture_source == "Video File":
        video_file = st.file_uploader("Upload a video file", type=["mp4"])
        if video_file is not None:
            st.video(video_file)
            st.write("Object Detection in Progress...")

            cap = cv2.VideoCapture(video_file.name)  # Corrected line
            assert cap.isOpened()

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            frame_container = st.empty()

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame = detector(frame)

                frame_container.image(processed_frame, channels="RGB", use_column_width=True)

                if cv2.waitKey(5) & 0xFF == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()

    elif capture_source == "Webcam":
        st.write("Object Detection in Progress...")
        cap = cv2.VideoCapture(0)

        frame_container = st.empty()

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = detector(frame)

            frame_container.image(processed_frame, channels="RGB", use_column_width=True)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
