from imageai.Detection.Custom import CustomObjectDetection
import cv2 as cv
import time

# Loading the model
detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("./models/model(loss=12.004).h5")
detector.setJsonPath("./data/json/detection_config.json")
detector.loadModel()

# Enabling camera access
vid = cv.VideoCapture("video.mp4")

# Variables storing box coordinates
min_pt = ()
max_pt = ()

frame_width = int(vid.get(3))
frame_height = int(vid.get(4))


def draw_box(img, coords, confidence):
    """
        To draw a box on the image given the hand coordinates
    """
    
    min_pt = (coords[0], coords[1])
    max_pt = (coords[2], coords[3])

    # Bounding box
    cv.rectangle(img, min_pt, max_pt, (0,0,255), 2)

    # Confidence probability
    cv.putText(img,f"Hand:{round(confidence,2)}%", min_pt,
    cv.FONT_HERSHEY_PLAIN,1,(0,255,0),1)

# FPS time variables
# prev_t,new_t = 0,0

# Video writer
out = cv.VideoWriter('video_detected.avi',cv.VideoWriter_fourcc('M','J','P','G'), 
50, (frame_width,frame_height))

frame_num = 0

# Running loop for detection
while vid.isOpened():
    _, frame = vid.read()
    frame_num+=1

    # new_t = time.time()

    # Detecting hands
    detections = detector.detectObjectsFromImage(input_image=frame, output_type="array", 
    input_type="array", minimum_percentage_probability=20, nms_treshold=0.2,
    display_object_name=False, display_percentage_probability=False)


    # Detecting hands from current 'frame'
    for detected in detections:
        if len(detected) and type(detected[0]) is dict:
            print(detected)

            # Drawing box around each detected hand
            for box in detected:
                pts = box.get("box_points")
                conf = box.get("percentage_probability")
                draw_box(frame,pts,conf)

        else:
            print("No hand")
    
    # fps = int(1/(new_t-prev_t))
    # prev_t = new_t
    # cv.putText(frame,f"FPS:{fps}",(30,30),
    # cv.FONT_HERSHEY_PLAIN,1.5,(0,0,255))
    
    # Write frame
    out.write(frame)
    print(frame_num)

    if cv.waitKey(1) == ord('q'):
        break

# Release resources
vid.release()
out.release()
cv.destroyAllWindows()