# distance_allclasses_spoken.py
# Unified distance estimator for all 80 COCO classes with spoken output (CPU mode).
# Requirements:
#   pip install opencv-contrib-python==4.5.3.56 numpy pyttsx3 imutils

import time
import cv2 as cv
import numpy as np
import imutils
import pyttsx3

# ----------------- USER SETTINGS -----------------
KNOWN_DISTANCE = 45  # inches, distance used to compute focal length
# Approx real-world widths (inches) for COCO classes (index 0..79) taken from project files
EACH_WIDTH = [16,12,12,12,12,12,12,12,12,12,15,16,10,18,5,8,15,20,14,20,22,18,20,20,10,20,12,3,12,12,12,15,10,8,5,5,10,10,10,3,4,4,1,1,1,3,2,3,4,3,4,1,2,5,5,8,16,20,6,25,20,10,10,10,2,2,9,3,12,12,10,10,12,8,11,12,4,16,12,4]
# Corresponding measured pixel widths from reference images (in the repo)
EACH_WIDTH_in_rf = [367,288,288,288,288,288,288,288,288,288,360,367,240,432,120,192,360,480,336,480,528,432,480,480,240,480,288,72,288,288,288,360,240,192,120,120,240,240,240,72,96,96,24,24,24,72,48,72,96,72,96,24,48,120,120,192,384,480,144,600,480,240,240,240,48,48,216,72,288,288,240,240,288,192,264,288,96,367,288,96]

# Model files (must be in the same folder)
CFG = 'yolov4-tiny.cfg'
WEIGHTS = 'yolov4-tiny.weights'
CLASSES = 'classes.txt'

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# Speech cooldown (seconds) per unique spoken message
SPEECH_COOLDOWN = 2.0
# --------------------------------------------------

# load class names
with open(CLASSES, 'r') as f:
    class_names = [c.strip() for c in f.readlines()]

# build a name->index map (COCO standard)
classes_map = {name: idx for idx, name in enumerate(class_names)}

# compute focal lengths for each class from reference pixel widths
focal_lengths = []
for i in range(len(EACH_WIDTH)):
    try:
        fl = (EACH_WIDTH_in_rf[i] * KNOWN_DISTANCE) / EACH_WIDTH[i]
    except Exception:
        fl = None
    focal_lengths.append(fl)

# initialize TTS
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# load YOLO network (CPU)
yoloNet = cv.dnn.readNet(WEIGHTS, CFG)
# force CPU backend & target so it works without CUDA
yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# helper: position (top/mid/bottom + left/center/right)
def position_of_box(rect, canvas_w=800, canvas_h=800):
    (x, y, w, h) = rect
    centerX = int((2*x + w) / 2)
    centerY = int((2*y + h) / 2)
    if centerX <= canvas_w/3:
        W_pos = "left"
    elif centerX <= (canvas_w/3 * 2):
        W_pos = "center"
    else:
        W_pos = "right"

    if centerY <= canvas_h/3:
        H_pos = "top"
    elif centerY <= (canvas_h/3 * 2):
        H_pos = "mid"
    else:
        H_pos = "bottom"
    return f"{H_pos} {W_pos}"

# distance formula
def distance_finder(focal_length, real_object_width, width_in_frame):
    if focal_length is None or width_in_frame == 0:
        return None
    return (real_object_width * focal_length) / width_in_frame

# small utility to normalize classid returned by model.detect
def to_int_classid(c):
    # model.detect may return e.g. array([[0], [67]]) or array([0,67])
    try:
        return int(np.array(c).flatten()[0])
    except:
        return int(c)

# track last spoken time for messages to avoid spam
last_spoken = {}

# capture
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Starting detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = imutils.resize(frame, width=800)

    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # ensure consistent shapes
    if len(classes) == 0:
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # classes might be Nx1 or N
    classes_flat = np.array(classes).flatten()
    scores_flat = np.array(scores).flatten()

    # collect spoken strings this frame
    frame_strings = []

    for cid, score, box in zip(classes_flat, scores_flat, boxes):
        cid = int(cid)

        # Only detect street-related objects
        if cid not in [0, 1, 2, 3, 5, 7, 9, 11, 15, 16]:
            continue  # skip to the next detection

        name = class_names[cid] if cid < len(class_names) else str(cid)
        (x, y, w, h) = box

        # draw box & label
        label = f"{name}: {score:.2f}"qq
        color = (0, 255, 0)
        cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # compute distance if we have a focal length and width
        if cid < len(focal_lengths) and focal_lengths[cid] is not None:
            real_w = EACH_WIDTH[cid]
            focal = focal_lengths[cid]
            distance = distance_finder(focal, real_w, w)
            if distance is not None:
                # show distance
                txt = f"Dis: {distance:.2f} in"
                cv.putText(frame, txt, (x, y + h + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                # prepare spoken string
                pos = position_of_box((x,y,w,h), canvas_w=800, canvas_h=800)
                spoken = f"{name} at {pos}, {int(round(distance))} inches"
                frame_strings.append(spoken)

    # speak unique strings but with cooldown
    now = time.time()
    for s in set(frame_strings):
        last = last_spoken.get(s, 0)
        if now - last >= SPEECH_COOLDOWN:
            try:
                engine.say(s)
                engine.runAndWait()
            except Exception as e:
                print("TTS error:", e)
            last_spoken[s] = now

    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
