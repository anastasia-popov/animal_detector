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
        try:
            self.model = YOLO(self.config['model_path'])
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
            results = self.model.track(frame, persist=True, conf=0.01, verbose=False)
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            return boxes, track_ids
        except Exception as e:
            logging.error(f"Error tracking objects: {e}")
            return [], []

    def calculate_movement(self, boxes, track_ids, track_history):
        max_distance = 0
        movement_cache = {}
        try:
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                if w * h > 300:
                    track = track_history[track_id]
                    track.append((float(x), float(y)))
                    if track_id in movement_cache:
                        dist = movement_cache[track_id]
                    else:
                        dist = np.linalg.norm(np.array(track[0]) - np.array(track[-1]))
                        movement_cache[track_id] = dist
                    max_distance = max(max_distance, dist)
        except Exception as e:
            logging.error(f"Error calculating movement: {e}")
        return max_distance

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

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                boxes, track_ids = self.track_objects(frame, track_history)
                max_distance = self.calculate_movement(boxes, track_ids, track_history)

                if max_distance > self.config['movement_threshold']:
                    movement_detected = True
                    break

            if movement_detected:
                logging.info(f'Movement detected')
                self.save_video()
                self.save_image(frame)
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

