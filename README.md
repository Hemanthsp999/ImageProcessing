# Assignment: Image Tracking and Re-tracking througout the video

## Logic Behind the model's inference pipeline: 

### 1. Object Detection

Objective: Identify and locate objects in each frame of the video.
Method:
Use a YOLO (You Only Look Once) model, which is a well-known real-time object detection algorithm.
Load a pre-trained YOLO model using a specified path.
Apply this model to each frame of the video to detect objects.
Output: For each object detected, you'll get a bounding box (the coordinates of the box that surrounds the object), a confidence score indicating how sure the model is about the detection, and a class label specifying the type of object.

### 2. Object Tracking

Objective: Track objects across multiple frames and assign a unique identifier to each object.
Method:
Use the DeepSORT (Deep Simple Online and Realtime Tracking) tracker, which helps in tracking objects over time and maintaining their identities.
Initialize the tracker with parameters that dictate how long to keep tracking an object and how to match objects between frames.
Update the tracker with the detected objects from the YOLO model for each frame.
Output: The tracker provides updated information about the objects being tracked, including their unique IDs, positions, and states.

### 3. Feature Extraction

Objective: Extract deep features from the detected objects to improve tracking accuracy.
Method:
Use a ResNet50 model, which is a powerful deep learning model pre-trained on a large dataset to recognize various features.
For each detected object, extract a Region of Interest (ROI) from the frame based on the object's bounding box.
Resize and preprocess this ROI to match the input requirements of the ResNet50 model.
Pass the preprocessed ROI through ResNet50 to get a feature vector that represents the object.
Output: A set of features for each tracked object, which helps in distinguishing between different objects and tracking them more accurately over time.

### 4. Similarity Calculation

Objective: Measure how similar two objects are, based on their bounding boxes and features.
Method:
Calculate the Intersection over Union (IoU) for the bounding boxes of two objects. IoU measures how much the bounding boxes overlap.
Compute the Euclidean distance between the feature vectors of two objects. This measures the difference in their deep features.
Combine these two metrics to create a similarity score that reflects both spatial overlap and feature similarity.
Output: A similarity score that indicates how closely related two objects are, which helps in deciding if a previously inactive track should be reactivated.

### 5. Handling Inactive Tracks

Objective: Manage objects that disappear from view and reappear later, ensuring they are tracked consistently.
Method:
Maintain a list of active and inactive track IDs.
Periodically check if any active tracks have not been detected for a while and mark them as inactive.
If an inactive track reappears, use the similarity score to determine if it should be reactivated with its old ID or assigned a new one.
Output: Updated lists of active and inactive tracks, ensuring consistent tracking of objects even if they leave and re-enter the frame.

## How Everything Works Together ?

### 1.Video Capture:

Capture video frames and place them into a queue for processing.

### 2.Model Inference:

Process each frame to detect objects using YOLO.
Track objects using DeepSORT and update their information.
Extract features from objects using ResNet50 to enhance tracking accuracy.
Draw tracking information on the frames and save the processed video.

### 3.Reactivating Tracks:

Compare inactive tracks with current detections to decide if they should be reactivated based on similarity scores.

# How to use the model

## Create a virtual-env

```bash
python3 -m venv 'your_env_name'
```

Activate the environment

```bash
source 'your_env_name/bin/activate'
```

## Get the dependencies of project

```bash
pip freeze > requirements.txt
```

Install the dependencies

```bash
pip install -r requirements.txt
```

## run the model

```bash
python main.py
```
