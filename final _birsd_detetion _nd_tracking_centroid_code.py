import cv2
import numpy as np
from time import time


# Twilio credentialsfrom time import time
from ultralytics import YOLO
import torch
from twilio.rest import Client
from collections import defaultdict


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
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.sms_sent = False
        self.model = YOLO(r"D:/project 156/project code and documents/working code/best2.pt")
        self.annotator = None
        self.start_time = 0
        self.end_time = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.frame_interval = 5
        self.time_threshold = 60
        self.last_movement_time = time()
        self.frames_without_movement = 0
        self.last_message_time = time()  # Initialize the last message time
        self.results = None  # Initialize results attribute
        self.track_history = defaultdict(lambda: [])

    def detect_movement(self, centroids):
        current_time = time()

        # Check if the message has already been sent in the last one minute
        if current_time - self.last_message_time < self.time_threshold:
            return False

        if not centroids:
            self.last_movement_time = current_time
            self.frames_without_movement = 0
        else:
            elapsed_time = current_time - self.last_movement_time
            if elapsed_time >= self.time_threshold:
                print(f"Sending SMS - Elapsed time without movement: {elapsed_time} seconds")
                self.last_message_time = current_time  # Update the last message time
                return True
            self.frames_without_movement += 1

            movement_detected = any(self.check_movement_radius(centroid) for centroid in centroids)
            if movement_detected:
                print(f"Sending SMS - Movement detected for centroids : {centroids}")
                self.last_message_time = current_time  # Update the last message time
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
        self.end_time = time()
        fps = 1 / np.round(self.end_time - self.start_time, 2)
        text = f'FPS: {int(fps)}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        gap = 10
        cv2.rectangle(im0, (20 - gap, 70 - text_size[1] - gap), (20 + text_size[0] + gap, 70 + gap), (0, 255, 0), -1)
        cv2.putText(im0, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    
    def plot_bboxes(self, results, im0):
        centroids = []
        self.annotator = Annotator(im0, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        confs = results[0].boxes.conf.cpu().tolist()  # Get the confidence values
        names = results[0].names
        conf_threshold = 0.2  # Set the confidence threshold
        for idx, (box, cls, conf) in enumerate(zip(boxes, clss, confs)):
            if conf >= conf_threshold:  # Check if the confidence is above the threshold
                
                centroid = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
                centroids.append(centroid)

                label = f"{names[int(cls)]}  {conf:.2f}"  # Display unique track ID

                color = (0, 0, 255)
                cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 3)
                
                # Set text color to white and background color to blue
                text_color = (255, 255, 255)
                bg_color = (0, 0, 255)
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(im0, (int(box[0]), int(box[1]) - text_size[1] - 5),
                          (int(box[0]) + text_size[0], int(box[1]) - 5), bg_color, -1)
                cv2.putText(im0, label, (int(box[0]), int(box[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

                # Store tracking history
                track = self.track_history[centroid]
                track.append(centroid)
                if len(track) > 30:
                    track.pop(0)

                # Plot tracks
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.circle(im0, (track[-1]), 7, (255,0, 0), -1)

        return im0, centroids

    def __call__(self):
        cap = cv2.VideoCapture('chicktest.mp4')
        assert cap.isOpened()

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        frame_count = 0

        while True:
            self.start_time = time()
            ret, im0 = cap.read()

            print(f"Frame read successful: {ret}")
            print(f"Frames without movement: {self.frames_without_movement}")

            if ret:
                results = self.predict(im0)
                im0, centroids = self.plot_bboxes(results, im0)

                if self.detect_movement(centroids):
                    if not self.sms_sent:
                        send_sms(to_number, twilio_number, len(centroids))
                        self.sms_sent = True
                    continue
                else:
                    self.sms_sent = False
                

                self.display_fps(im0)
                cv2.imshow('YOLOv8 Detection', im0)
                frame_count += 1

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


# Instantiate the ObjectDetection class
detector = ObjectDetection(capture_index=0)

# Don't call the __call__ method here, as it's automatically called when using ()
# detector()
