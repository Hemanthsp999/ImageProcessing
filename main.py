import cv2
import os
import time
import numpy as np
import threading
import queue
from collections import deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


# YOLO model with Custom_Dataset as Training data
def load_model(model_path):
    return YOLO(model_path)


# Initialize the DeepSort to track the object with unique ids
def initialize_tracker(max_age=5000, n_init=2, max_iou_distance=0.4):
    return DeepSort(max_age=max_age, n_init=n_init, max_iou_distance=max_iou_distance)


# Process the frames to predict the result
def process_detections(results):
    # append the predicted results to detections list if its confidence > 0.7
    detections = []
    for result in results:
        for box in result.boxes:
            label_index = int(box.cls[0])
            if label_index not in [0, 1]:
                continue
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            xmin, ymin, xmax, ymax = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
            width = xmax - xmin
            height = ymax - ymin
            conf = float(box.conf[0])
            if conf > 0.7:
                detections.append(
                    [[xmin, ymin, width, height], conf, label_index])
    return detections  # returns the lists of min, max, width, height, confidence and label_index


# Loads pre-trained ResNet50 model wihout the top classification layer, using it for feature extraction
def load_pretrained_model():
    return ResNet50(weights='imagenet', include_top=False, pooling='avg')


# call Pretrained model
pre_trained_model = load_pretrained_model()


# This function updates the track history for each object, storing recent bounding boxes and deep feature extracted using ResNet50
def update_track_history(track_id, bbox, frame, track_history):
    # Validate bounding box
    x_min, y_min, width, height = map(int, bbox)
    frame_height, frame_width, _ = frame.shape

    # Ensure bbox is within the frame boundaries
    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
    if x_min + width > frame_width:
        width = frame_width - x_min
    if y_min + height > frame_height:
        height = frame_height - y_min

    # Check if the bounding box has a positive size
    if width <= 0 or height <= 0:
        print(f"Invalid bounding box for track {track_id}: {bbox}")
        return  # Skip this bounding box

    if track_id not in track_history:
        track_history[track_id] = {'bboxes': deque(maxlen=5), 'features': None}

    track_history[track_id]['bboxes'].append(bbox)

    # Extract deep features using ResNet50
    roi = frame[y_min:y_min+height, x_min:x_min+width]

    # Resize ROI to fit the input size of ResNet50 (224x224)
    roi_resized = cv2.resize(roi, (224, 224))
    roi_resized = preprocess_input(roi_resized)  # Preprocess for ResNet50

    # Expand dimensions to match the input shape (1, 224, 224, 3)
    roi_resized = np.expand_dims(roi_resized, axis=0)

    # Get deep features
    deep_features = pre_trained_model.predict(roi_resized)
    deep_features = deep_features.flatten()

    # If no previous features, initialize with the current one, else update
    if track_history[track_id]['features'] is None:
        track_history[track_id]['features'] = deep_features
    else:
        track_history[track_id]['features'] = 0.7 * \
            track_history[track_id]['features'] + 0.3 * deep_features


# combines IoU and feature vector to calculate similarity score between two tracks
def calculate_similarity(bbox1, features1, bbox2, features2):
    iou = calculate_iou(bbox1, bbox2)
    feature_similarity = np.linalg.norm(features1 - features2)
    return 0.7 * iou + 0.3 * (1 / (1 + feature_similarity))


# Draw bounding boxes and labels for each confirmed track on the resized frame
def draw_tracks(frames_resize, tracks, track_history):
    for track in tracks:
        if track.is_confirmed() and track.time_since_update <= 1:
            bbox = track.to_tlbr()
            track_id = track.track_id
            label = "Child" if track.get_det_class() == 0 else "Therapist"
            color = (255, 150, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_thickness = 2
            font_scale = 1
            cv2.rectangle(frames_resize, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(frames_resize, f"{label} ID:{track_id}",
                        (int(bbox[0]), int(bbox[1]) - 10),
                        font, font_scale, (0, 0, 0), font_thickness)
            update_track_history(track_id, bbox, frames_resize, track_history)


# calculates the IoU between two boxes.IoU measures the overlap between two boxes, which is crucial for tracking
# remember your custom_data does not contain any overlap images, so it will affect model prediction in future
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea -
                            interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou


# Tracks that have not been detected in current frames are marked as inactive and stored for possible reactivation
def manage_inactive_tracks(tracks, inactive_tracks, track_history):
    active_ids = {track.track_id for track in tracks if track.is_confirmed()}
    for track_id in list(track_history.keys()):
        if track_id not in active_ids and track_id not in inactive_tracks:
            inactive_tracks.append(track_id)
            print(f"Track {track_id} marked as inactive.")


# converts to bounding box format i.e [x, y, w, h]
def convert_bbox_format(bbox):
    x_min, y_min, x_max, y_max = bbox
    return [x_min, y_min, x_max - x_min, y_max - y_min]


# process frames in realtime, utilizing YOLO model for detections and DeepSort for traking
def handle_tracks(tracks, inactive_tracks, track_history, frame):
    print(f"current tracks: {[track.track_id for track in tracks]}")
    print(f"Inactive tracks: {inactive_tracks}")
    active_track_ids = {track.track_id for track in tracks}

    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id
            bbox = convert_bbox_format(track.to_tlbr())

            if track_id in inactive_tracks:
                last_bbox = track_history[track_id]['bboxes'][-1]
                last_hist = track_history[track_id]['features']
                similarity = calculate_similarity(
                    last_bbox, last_hist, bbox, track_history[track_id]['features']
                )
                print(
                    f"Reactivation similarity for track {track_id}: {similarity}")

                if similarity > 0.8:
                    inactive_tracks.remove(track_id)
                    print(f"Track {track_id} reactivated.")
                else:
                    print(f"Track {track_id} is still inactive.")
            else:
                reactivated = False
                for inactive_track_id in inactive_tracks:
                    last_bbox = track_history[inactive_track_id]['bboxes'][-1]
                    last_hist = track_history[inactive_track_id]['features']
                    similarity = calculate_similarity(
                        last_bbox, last_hist, bbox, track_history[inactive_track_id]['features']
                    )
                    reactivated = True
                    print("reactivation status", reactivated)
                    print(
                        f"Reactivation similarity for inactive track {inactive_track_id}: {similarity}")

                    if similarity > 0.8:
                        track_id = inactive_track_id
                        inactive_tracks.remove(track_id)
                        reactivated = True
                        print(f"Track {track_id} reactivated with old ID.")
                        break

                if not reactivated:
                    print(f"New track_id {track_id} created.")

                update_track_history(track_id, bbox, frame, track_history)

    for track_id in list(track_history.keys()):
        if track_id not in active_track_ids and track_id not in inactive_tracks:
            inactive_tracks.append(track_id)
            print(f"Track {track_id} marked as inactive.")


# This caputres the video using cv2 and breaks into frames for furthur process
def video_capture_thread(video_path, frame_queue, fps_queue):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)  # Capture FPS of the video
    fps_queue.put(video_fps)
    print(f"original video fps {video_fps}")
    # Skip frames to match real-time if necessary
    # frame_interval = int(video_fps / 15)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_queue.full():
            time.sleep(9)
        else:
            frame_queue.put(frame)
    cap.release()
    frame_queue.put(None)


# another thread that process the frames and write into mp4 file as output
def model_inference_thread(frame_queue, model, tracker, track_history, inactive_tracks, output_video_path, fps_queue):
    # get the fps_queue which is passed from video_capture_thread
    video_fps = fps_queue.get()
    print(f"using video FPS: {video_fps}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, video_fps, (520, 520))

    frame_interval = 1.0 / video_fps

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame is None:
                break

            # Process time difference to sync frame rate

            start_time = time.time()

            # Resize frame for inference (Consider reducing size for speed)
            frames_resize = cv2.resize(frame, (520, 520))

            results = model.predict(frames_resize)
            detections = process_detections(results)

            tracks = tracker.update_tracks(detections, frame=frames_resize)
            draw_tracks(frames_resize, tracks, track_history)
            manage_inactive_tracks(tracks, inactive_tracks, track_history)
            handle_tracks(tracks, inactive_tracks,
                          track_history, frame)

            out.write(frames_resize)
            elapsed_time = time.time() - start_time
            time_to_wait = frame_interval - elapsed_time
            if time_to_wait > 0:
                time.sleep(time_to_wait)

            # Exit loop if 'q' is pressed
            if cv2.waitKey(10) == ord('q'):
                break
    out.release()
    cv2.destroyAllWindows()


# main function it calls two thread i.e one for video_capture_thread that captures the frames and
# another is model_inference_thread this process the predicted result and writes to output
# this makes the model to work efficiently in less time
def main(video_path, model_path, output_video_path):
    if not (os.path.exists(video_path)) and (os.path.exists(model_path)):
        raise OSError(
            "File Path not exists or wrong path is mentioned( Check once)")

    # Pass frames in queue to overcome the frames drops issue
    frame_queue = queue.Queue(maxsize=100)
    fps_queue = queue.Queue(maxsize=100)

    # Load the model
    model = load_model(model_path)
    # Initialize the tracker
    tracker = initialize_tracker()

    # Track the frame history if i.e if frame already created then it tracks it for furthur process
    # Through mapping the track_id by using key as frames
    track_history = {}
    # If a person is out of the frame then it assigns track_id to inactive_state till he gets in
    inactive_tracks = []

    # Use Thread to make video_process and frames processing efficiently
    t1 = threading.Thread(target=video_capture_thread,
                          args=(video_path, frame_queue, fps_queue))
    t1.start()
    t2 = threading.Thread(target=model_inference_thread, args=(
        frame_queue, model, tracker, track_history, inactive_tracks, output_video_path, fps_queue))

    t2.start()

    t1.join()
    t2.join()


if __name__ == "__main__":
    # Give your video path
    video_path = "test_videos/TestVideo_11.mp4"
    # Give Your model path i.e best.pt
    model_path = "runs/detect/train/weights/best.pt"
    # Give filepath to store the output
    output_video_path = "output_videos/output_11.mp4"
    main(video_path, model_path, output_video_path)
