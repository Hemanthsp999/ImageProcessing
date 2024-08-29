from ultralytics import YOLO
import cv2

# load the model
model = YOLO("runs/detect/train/weights/best.pt")
getVideo = cv2.VideoCapture("test_videos/TestVideo_10.mp4")

prediction = []

while getVideo.isOpened():

    # capture the frames
    ret, frames = getVideo.read()
    if not ret:
        break

    frames = cv2.resize(frames, (720, 380), interpolation=cv2.INTER_CUBIC)

    # Preict the frames
    results = model.predict(frames)

    for frame in results:
        if hasattr(frame, 'boxes'):
            for box in frame.boxes:
                label_index = int(box.cls[0])

                if label_index not in [0, 1]:
                    continue

                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                label_index = int(box.cls[0])
                conf = float(box.conf[0])
                label_name = "Child" if label_index == 0 else "Therapist"
                print("Label Index is ", label_index)

                cv2.rectangle(frames, (xyxy[0], xyxy[1]),
                              (xyxy[2], xyxy[3]), (255, 0, 0), 2)

                cv2.putText(frames, f"{label_name} {conf: .2f}", (
                    xyxy[0], xyxy[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

                label = model.names[label_index]
                prediction.append({"label": label, "conf": conf})

    cv2.imshow("frame", frames)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

getVideo.release()

cv2.destroyAllWindows()
