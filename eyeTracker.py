from statistics import mode

import cv2
from keras.models import load_model
import numpy as np
import screeninfo
import dlib
import time
import requests
import math
import copy

screen_id = 0
screen = screeninfo.get_monitors()[screen_id]
width, height = screen.width, screen.height
image = np.ones((height, width, 3), dtype=np.float32)
image[:10, :10] = 0  # black at top-left corner
image[height - 10:, :10] = [1, 0, 0]  # blue at bottom-left
image[:10, width - 10:] = [0, 1, 0]  # green at top-right
image[height - 10:, width - 10:] = [0, 0, 1]  # red at bottom-right

BLINK_RATIO_THRESHOLD = 5

def avg_helper(segment):
    fleft = []
    fright = []
    sleftx = []
    slefty = []
    srightx = []
    srighty = []
    output = []
    for i in range(len(segment)):
        fleft.append(segment[i][0])
        fright.append(segment[i][1])
        for ind in range(len(fleft)):
            sleftx.append(fleft[ind][0])
            slefty.append(fleft[ind][1])
            srightx.append(fright[ind][0])
            srighty.append(fright[ind][1])
    output.append(sum(sleftx) / len(sleftx))
    output.append(sum(slefty) / len(slefty))
    output.append(sum(srightx) / len(srightx))
    output.append(sum(srighty) / len(srighty))
    return output

def averages(lister):
    first = lister[slice(0, int(len(lister) / 4))]
    second = lister[slice(int(len(lister) / 4), int(len(lister) / 2))]
    third = lister[slice(int(len(lister) / 2), int(3 * (len(lister) / 4)))]
    fourth = lister[slice(int(3 * (len(lister) / 4)), int(len(lister)))]
    output1 = avg_helper(first)
    output2 = avg_helper(second)
    output3 = avg_helper(third)
    output4 = avg_helper(fourth)
    ofirst = []
    osecond = []
    ofirst.extend([output1, output2])
    osecond.extend([output3, output4])
    return ofirst, osecond

def midpoint(point1, point2):
    return (point1.x + point2.x) / 2, (point1.y + point2.y) / 2


def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def get_blink_ratio(eye_points, facial_landmarks):
    # loading all the required points
    corner_left = (facial_landmarks.part(eye_points[0]).x,
                   facial_landmarks.part(eye_points[0]).y)
    corner_right = (facial_landmarks.part(eye_points[3]).x,
                    facial_landmarks.part(eye_points[3]).y)

    center_top = midpoint(facial_landmarks.part(eye_points[1]),
                          facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]),
                             facial_landmarks.part(eye_points[4]))

    # calculating distance
    horizontal_length = euclidean_distance(corner_left, corner_right)
    vertical_length = euclidean_distance(center_top, center_bottom)

    ratio = horizontal_length / vertical_length

    return ratio

detector = dlib.get_frontal_face_detector()

# -----Step 4: Detecting Eyes using landmarks in dlib-----
predictor = dlib.shape_predictor("D:/face_classification-master/src/models/shape_predictor_68_face_landmarks.dat")
# these landmarks are based on the image above
left_eye_landmarks = [36, 37, 38, 39, 40, 41]
right_eye_landmarks = [42, 43, 44, 45, 46, 47]

waterstart = time.time()
eyestrainstart = time.time()

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

# parameters for loading data and images
detection_model_path = 'D:/face_classification-master/trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = 'D:/face_classification-master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask


def contouring(thresh, mid, img, right=False):
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    try:
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        coords = [cx, cy]
        return coords

    except:
        pass


file = 'D:/face_classification-master/src/models/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(file)

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

video_capture = cv2.VideoCapture(0)
ret, img = video_capture.read()
thresh = img.copy()
cv2.namedWindow('window_frame')

kernel = np.ones((9, 9), np.uint8)
def nothing(x):
    pass
cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
fineye = [300, 300, 200, 200]
blinked=False
f=True
first_iter = False
blink_counter=0
setup_counter=0
x_offset = y_offset = 50
top_cord = []
right_cord = []
bottom_cord=[]
left_cord=[]
added = True
allow = False
text=False
match=False
nextScreen=False
homePage=False
initial=True
state=-1
tblr=-1
while True:
    plus_sign = cv2.imread("D:/CraneRobot/images/plussign.png")
    background = cv2.imread("D:/CraneRobot/images/greyback.jpg")
    uppic = cv2.imread("D:/CraneRobot/images/up.png")
    downpic = cv2.imread("D:/CraneRobot/images/down.png")
    rightpic = cv2.imread("D:/CraneRobot/images/right.png")
    leftpic = cv2.imread("D:/CraneRobot/images/left.png")
    ret, img = video_capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces, _, _ = detector.run(image=frame, upsample_num_times=0,
                               adjust_threshold=0.0)
    for face in faces:
        landmarks = predictor(frame, face)
        left_eye_ratio = get_blink_ratio(left_eye_landmarks, landmarks)
        right_eye_ratio = get_blink_ratio(right_eye_landmarks, landmarks)
        blink_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blink_ratio > BLINK_RATIO_THRESHOLD:
            blinked=True
            if f:
                first = time.time()
                f=False
                blink_counter += 1
            else:
                second = time.time()
                print(second - first)
                if second - first >= 1:
                    blink_counter += 1
                    first = time.time()
            print(blink_counter)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = eye_on_mask(mask, left)
        mask = eye_on_mask(mask, right)
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        # fin = copy.deepcopy(eyes_gray)
        # print(np.amax(fin), np.amin(fin))
        # mask = fin >= 250
        # fin[mask] = 255
        # rmask = fin < 250
        # fin[rmask] = 0
        # cv2.imshow('mask', fin)
        threshold = 115
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)  # 1
        thresh = cv2.dilate(thresh, None, iterations=4)  # 2
        thresh = cv2.medianBlur(thresh, 3)  # 3
        thresh = cv2.bitwise_not(thresh)
        lefteye = contouring(thresh[:, 0:mid], mid, img)
        righteye = contouring(thresh[:, mid:], mid, img, True)

    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    added_image = cv2.addWeighted(img, 0.4, bgr_image, 0.5, 0)
    cv2.imshow('Eye Tracking', added_image)


    if blink_counter == 0 or blink_counter == 1:
        cv2.putText(background, "Calibration Stage", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(background,
                    "Look at the plus sign that slides around the screen and follow it to capture your gaze.",
                    (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(background, "Blink twice to start the calibration process!", (10, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)



    if blink_counter == 2:
        allow = True
        first_iter = True

    if allow:
        if first_iter:
            background[y_offset:y_offset + plus_sign.shape[0], x_offset:x_offset + plus_sign.shape[1]] = plus_sign
            first_iter = False
        else:
            if y_offset == 50 and x_offset <= 1850:
                top_cord.append([lefteye, righteye])
                x_offset += 20
                background[y_offset:y_offset + plus_sign.shape[0], x_offset:x_offset + plus_sign.shape[1]] = plus_sign
                time.sleep(.1)
            elif x_offset == 1870 and y_offset<=1010:
                right_cord.append([lefteye, righteye])
                y_offset+=20
                background[y_offset:y_offset + plus_sign.shape[0], x_offset:x_offset + plus_sign.shape[1]] = plus_sign
                time.sleep(.1)
            elif y_offset==1030 and x_offset>=70:
                bottom_cord.append([lefteye, righteye])
                x_offset-=20
                background[y_offset:y_offset + plus_sign.shape[0], x_offset:x_offset + plus_sign.shape[1]] = plus_sign
                time.sleep(.1)
            elif x_offset == 50 and y_offset>=70:
                left_cord.append([lefteye, righteye])
                y_offset -= 20
                background[y_offset:y_offset + plus_sign.shape[0], x_offset:x_offset + plus_sign.shape[1]] = plus_sign
                time.sleep(.1)
                if x_offset == 50 and y_offset==50:
                    allow = False
                    text=True

    if homePage:
        if initial:
            fixed_blink_count=4
            blink_counter=4
            initial = False
        if blink_counter % fixed_blink_count == 0:
            state=0
            cv2.putText(background, "DRIVE", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(background,
                        "Drive the robot using up, down, left, ",
                        (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(background, "and right eye movement!", (10, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            x_offset_up = 935
            y_offset_up = 50
            background[y_offset_up:y_offset_up + uppic.shape[0], x_offset_up:x_offset_up + uppic.shape[1]] = uppic
            cv2.putText(background,
                        "Forward",
                        (900, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            x_offset_down = 935
            y_offset_down = 977
            background[y_offset_down:y_offset_down + downpic.shape[0], x_offset_down:x_offset_down + downpic.shape[1]] = downpic
            cv2.putText(background,
                        "Backward",
                        (890, 957), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            x_offset_right = 1770
            y_offset_right = 515
            background[y_offset_right:y_offset_right + rightpic.shape[0], x_offset_right:x_offset_right + rightpic.shape[1]] = rightpic
            cv2.putText(background,
                        "Right",
                        (1680, 545), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            x_offset_left = 50
            y_offset_left = 515
            background[y_offset_left:y_offset_left + leftpic.shape[0], x_offset_left:x_offset_left + leftpic.shape[1]] = leftpic
            cv2.putText(background,
                        "Left",
                        (110, 545), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        elif blink_counter % fixed_blink_count == 1:
            state=1
            cv2.putText(background, "ROTATE", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(background,
                        "Rotate the robot's crane using left",
                        (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(background, "and right eye movement!", (10, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            x_offset_right = 1770
            y_offset_right = 515
            background[y_offset_right:y_offset_right + rightpic.shape[0],
            x_offset_right:x_offset_right + rightpic.shape[1]] = rightpic
            cv2.putText(background,
                        "Right",
                        (1680, 545), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            x_offset_left = 50
            y_offset_left = 515
            background[y_offset_left:y_offset_left + leftpic.shape[0],
            x_offset_left:x_offset_left + leftpic.shape[1]] = leftpic
            cv2.putText(background,
                        "Left",
                        (110, 545), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        elif blink_counter % fixed_blink_count == 2:
            state=2
            cv2.putText(background, "RAISE/LOWER", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(background,
                        "Raise or lower the robot's arm using up",
                        (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(background, "and down eye movement!", (10, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            x_offset_up = 935
            y_offset_up = 50
            background[y_offset_up:y_offset_up + uppic.shape[0], x_offset_up:x_offset_up + uppic.shape[1]] = uppic
            cv2.putText(background,
                        "Raise",
                        (915, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            x_offset_down = 935
            y_offset_down = 977
            background[y_offset_down:y_offset_down + downpic.shape[0],
            x_offset_down:x_offset_down + downpic.shape[1]] = downpic
            cv2.putText(background,
                        "Lower",
                        (905, 957), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        elif blink_counter % fixed_blink_count == 3:
            state=3
            cv2.putText(background, "OPEN/CLOSE", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(background,
                        "Open or close the robot's claw using up",
                        (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(background, "and down eye movement!", (10, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            x_offset_up = 935
            y_offset_up = 50
            background[y_offset_up:y_offset_up + uppic.shape[0], x_offset_up:x_offset_up + uppic.shape[1]] = uppic
            cv2.putText(background,
                        "Open",
                        (915, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            x_offset_down = 935
            y_offset_down = 977
            background[y_offset_down:y_offset_down + downpic.shape[0],
            x_offset_down:x_offset_down + downpic.shape[1]] = downpic
            cv2.putText(background,
                        "Close",
                        (905, 957), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)




    if nextScreen:
        cv2.putText(background, "Calibration Finished!", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(background,
                    "You can now control your robot! Switch between modes by blinking and control the robot through your gaze.",
                    (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(background, "Starting app...", (10, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        time.sleep(5)
        nextScreen=False
        homePage=True

    if allow==False and text==True:
        top_cal_1, top_cal_2 = averages(top_cord)
        right_cal_1, right_cal_2 = averages(right_cord)
        bottom_cal_1, bottom_cal_2 = averages(bottom_cord)
        left_cal_1, left_cal_2 = averages(left_cord)
        # print("\n", top_cord, "\n", right_cord, "\n", bottom_cord, "\n", left_cord, "\n")
        text=False
        match=True
        nextScreen=True

    if match:
        if (top_cal_1[0][0] <= lefteye[0] <= top_cal_2[1][0] or top_cal_1[0][0] >= lefteye[0] >= top_cal_2[1][0]) and (top_cal_1[0][2] <= righteye[0] <= top_cal_2[1][2] or top_cal_1[0][2] >= righteye[0] >= top_cal_2[1][2]) and (top_cal_1[0][1] <= lefteye[1] <= top_cal_2[1][1] or top_cal_1[0][1] >= lefteye[1] >= top_cal_2[1][1]) and (top_cal_1[0][3] <= righteye[1] <= top_cal_2[1][3] or top_cal_1[0][3] >= righteye[1] >= top_cal_2[1][3]):
            tblr=0
        elif (right_cal_1[0][0] <= lefteye[0] <= right_cal_2[1][0] or right_cal_1[0][0] >= lefteye[0] >= right_cal_2[1][0]) and (right_cal_1[0][2] <= righteye[0] <= right_cal_2[1][2] or right_cal_1[0][2] >= righteye[0] >= right_cal_2[1][2]) and (right_cal_1[0][1] <= lefteye[1] <= right_cal_2[1][1] or right_cal_1[0][1] >= lefteye[1] >= right_cal_2[1][1]) and (right_cal_1[0][3] <= righteye[1] <= right_cal_2[1][3] or right_cal_1[0][3] >= righteye[1] >= right_cal_2[1][3]):
            tblr=1
        elif (bottom_cal_1[0][0] <= lefteye[0] <= bottom_cal_2[1][0] or bottom_cal_1[0][0] >= lefteye[0] >= bottom_cal_2[1][0]) and (bottom_cal_1[0][2] <= righteye[0] <= bottom_cal_2[1][2] or bottom_cal_1[0][2] >= righteye[0] >= bottom_cal_2[1][2]) and (bottom_cal_1[0][1] <= lefteye[1] <= bottom_cal_2[1][1] or bottom_cal_1[0][1] >= lefteye[1] >= bottom_cal_2[1][1]) and (bottom_cal_1[0][3] <= righteye[1] <= bottom_cal_2[1][3] or bottom_cal_1[0][3] >= righteye[1] >= bottom_cal_2[1][3]):
            tblr=2
        elif (left_cal_1[0][0] <= lefteye[0] <= left_cal_2[1][0] or left_cal_1[0][0] >= lefteye[0] >= left_cal_2[1][0]) and (left_cal_1[0][2] <= righteye[0] <= left_cal_2[1][2] or left_cal_1[0][2] >= righteye[0] >= left_cal_2[1][2]) and (left_cal_1[0][1] <= lefteye[1] <= left_cal_2[1][1] or left_cal_1[0][1] >= lefteye[1] >= left_cal_2[1][1]) and (left_cal_1[0][3] <= righteye[1] <= left_cal_2[1][3] or left_cal_1[0][3] >= righteye[1] >= left_cal_2[1][3]):
            tblr=3
        else:
            tblr=4

        if state == 1:
            if tblr == 0 or tblr == 2:
                tblr = 4
        elif state == 2:
            if tblr == 1 or tblr == 3:
                tblr = 4
        elif state == 3:
            if tblr == 1 or tblr == 3:
                tblr = 4
        pos = {'device': 2, 'tblr': tblr}
        response = requests.post('http://saranggoel.pythonanywhere.com/', json=pos)

    blinked=False

    window_name = 'Interface'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, background)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()