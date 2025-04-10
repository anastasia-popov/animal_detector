import os
import cv2
import re
import yaml
import numpy as np
import logging
import shutil
import argparse
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO, settings

class MovementDetector:
    def __init__(self, input_video, config_path="config.yaml"):
        self.load_config(config_path)
        self.input_video = input_video
        self.id_class_dict = {}
        self.is_night = False
        try:
            self.model = YOLO(self.config['day_model_path'])
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}")
            raise
        settings.update({"runs_dir": "yolo"})
        logging.basicConfig(level=logging.INFO, format=self.config['log_format'])

    def load_config(self, config_path):
        try:
            with open(config_path, "r") as file:
                self.config = yaml.safe_load(file)
        except Exception as e:
            logging.error(f"Failed to load config file: {e}")
            raise

    @staticmethod
    def get_capture_date(file_path):
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            dt = datetime.fromtimestamp(os.path.getctime(file_path))
            return dt.strftime("%Y-%m-%d"), dt.strftime("%H.%M.%S")
        except Exception as e:
            logging.error(f"Error retrieving capture date for {file_path}: {e}")
            return "unknown_date", "unknown_time"
    
    def set_daytime(self, frame):
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation_channel = hsv_image[:, :, 1]
        mean_saturation = saturation_channel.mean()
        if mean_saturation < 5:
            self.is_night = True
            self.model = YOLO(self.config['night_model_path'])
            logging.info("Using night model")
        else:
            logging.info("Using day model")

    def save_video(self):
        try:
            predicted_dir = os.path.join(self.config['output_dir'], self.get_capture_date(self.input_video)[0])
            os.makedirs(predicted_dir, exist_ok=True)
            predicted_video = os.path.join(predicted_dir, os.path.basename(self.input_video))
            if not os.path.isfile(predicted_video):
                shutil.copy(self.input_video, predicted_video)
        except Exception as e:
            logging.error(f"Failed to save video {self.input_video}: {e}")

    def save_image(self, frame):
        try:
            img_file_name = re.sub(r'\.\w+$', '', os.path.basename(self.input_video)) + '.jpg'
            predicted_dir = os.path.join(self.config['output_dir'], self.get_capture_date(self.input_video)[0])
            os.makedirs(predicted_dir, exist_ok=True)
            predicted_img = os.path.join(predicted_dir, img_file_name)
            cv2.imwrite(predicted_img, frame)
        except Exception as e:
            logging.error(f"Failed to save image: {e}")

    def track_objects(self, frame, track_history):
        try:
            boxes, track_ids = [], []
            annotated_frame = frame
            results = self.model.track(frame, persist=True, conf=0.01, verbose=False)
            if not(results[0].boxes.id is None):
               boxes = results[0].boxes.xywh.cpu()
               track_ids = results[0].boxes.id.int().cpu().tolist()
               annotated_frame = results[0].plot()
               cls = results[0].boxes.cls.int().cpu().tolist()
               for track_id, box_class in zip(track_ids, cls):
                   if not(track_id in self.id_class_dict):
                       self.id_class_dict[track_id] = box_class
            return boxes, track_ids, annotated_frame
        except Exception as e:
            logging.error(f"Error tracking objects: {e}")
            return [], [], annotated_frame

    def calculate_movement(self, boxes, track_ids, track_history):
        max_distance = 0
        max_track = []
        max_track_object = ""
        object_box = [0,0,0,0]
        try:
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                if w * h > 300:
                    track = track_history[track_id]
                    track.append((float(x), float(y)))
                    dist = np.linalg.norm(np.array(track[0]) - np.array(track[-1]))
                    if dist > max_distance:
                        max_distance = dist
                        max_track = track_history[track_id]
                        max_track_object =  self.model.names[self.id_class_dict[track_id]]
                        object_box = box
        except Exception as e:
            logging.error(f"Error calculating movement: {e}")
        return max_distance, max_track, max_track_object, object_box

    def process_video(self):
        try:
            logging.info(f'Processing {self.input_video} {os.path.getsize(self.input_video) / (1024 * 1024):.0f} Mb')
            start = datetime.now()
            cap = cv2.VideoCapture(self.input_video)
            if not cap.isOpened():
                logging.error(f"Failed to open video {self.input_video}")
                return

            track_history = defaultdict(lambda: [])
            movement_detected = False
            tracked_object = ""
            object_box = [0,0,0,0]
            frame_num = 0

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                frame_num +=1

                if frame_num < 2:
                    self.set_daytime(frame)
                    
                if frame_num % int(self.config['frame_rate']):
                    continue

                boxes, track_ids, annotated_frame  = self.track_objects(frame, track_history)
                max_distance, max_track, tracked_object, object_box = self.calculate_movement(boxes, track_ids, track_history)

                if max_distance > self.config['movement_threshold']:
                    movement_detected = True
                    break

            if movement_detected:
                logging.info(f'Movement detected: {tracked_object}')
                self.save_video()
                points = np.hstack(max_track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 0, 255), thickness=10)
                self.save_image(annotated_frame)
            
            os.remove(self.input_video)
        except Exception as e:
            logging.error(f"Error processing video {self.input_video}: {e}")
        finally:
            cap.release()
            logging.info(f'Processing time: {datetime.now() - start}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', action='store', default='config.yaml')
    parser.add_argument('--input_video', dest='input_video', action='store')
    args = parser.parse_args()

    detector = MovementDetector(input_video=args.input_video, config_path=args.config)
    detector.process_video()

