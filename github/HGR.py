import numpy as np
import cv2
from .handdata






background = None
hand = None 

frame_elapsed = 0
FRAME_HEIGHT = 300
FRAME_WIDTH = 400

CALIBRATION_TIME = 30
BG_WEIGHT = 0.5 
OBJ_THRESHOLD = 110

def write_on_image(frame):

    text = 'Searching...'

    if frame_elapsed < CALIBRATION_TIME:
        text = "Calibrating..."
    elif hand == None or hand.isInFrame == False:
        text ="No hand detected"


    cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255),1,cv2.LINE_AA)

    cv2.rectangle(frame,(region_left,region_top),(region_right,region_bottom),(255,255,255) , 2)

def get_region(frame):
    region = frame[region_top:region_bottom, region_left:region_right]
    region = cv2.cvtColor(region,cv2.COLOR_BGR2GRAY)
    region = cv2.GaussianBlur(region,(7,7),0)

    return region


def get_average(region):

    global background

    if background is None:
        background = region.copy().astype("float")
        return
    cv2.accumulateWeighted(region,background,BG_WEIGHT)


def segment(region):

    global hand
    diff = cv2.absdiff(background.astype(np.uint8),region)

    thresholded_region = cv2.threshold(diff,OBJ_THRESHOLD,255, cv2.THRESH_BINARY)[1]

    (contours,_) = cv2.findContours(thresholded_region.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        if hand is not None:
            hand.isInFrame = False
        return 
    else:
        if hand is not None:
            hand.isInFrame = True
        segmented_region = max(contours, key = cv2.contourArea)
        return (thresholded_region,segmented_region)
    


def get_hand_data(thresholded_image,segmented_image):
    global hand

    convexHull = cv2.convexHull(segmented_image)

    top    = tuple(convexHull[convexHull[:,:,1].argmin()][0])
    bottom = tuple(convexHull[convexHull[:,:,1].argmax()][0])
    left   = tuple(convexHull[convexHull[:,:,0].argmin()][0])
    right  = tuple(convexHull[convexHull[:,:,0].argmax()][0])

    centerX = int((left[0] + right[0]) / 2)

    if hand == None:
        hand = HandData(top,bottom,left,right,centerX)
    else:
        hand.update(top,bottom,left,right)

    if frame_elapsed % 6 == 0:
        hand.check_for_waving(centerX)

region_top = 0
region_bottom = int(2 * FRAME_HEIGHT/3)
region_left = int(FRAME_WIDTH/2)
region_right = FRAME_WIDTH

frame_elapsed = 0 

capture = cv2.VideoCapture(0)

while(True):

    ret, frame = capture.read()
    
    frame = cv2.resize(frame, (FRAME_WIDTH,FRAME_HEIGHT))

    frame = cv2.flip(frame, 1)

    region = get_region(frame)
    if frame_elapsed < CALIBRATION_TIME:
        get_average(region)
    else:
        region_pair = segment(region)
        if region_pair is not None:
            (threshold_region,segmented_region) = region_pair
            cv2.drawContours(region,[segmented_region], -1, (255,255,255))
            cv2.imshow("Segmented Image", region)

            get_hand_data(threshold_region,segmented_region)


    write_on_image(frame)

    cv2.imshow("Camera Input",frame)

    frame_elapsed += 1

    if (cv2.waitKey(1) & 0xFF == ord('x')):
        break
        
    if (cv2.waitKey(1) & 0xFF == ord('r')):
        frames_elapsed = 0 

capture.release()
cv2.destroyAllWindows() 