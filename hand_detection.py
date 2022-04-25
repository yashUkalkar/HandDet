from imageai.Detection.Custom import CustomObjectDetection
import cv2 as cv
# import time

# File paths
model_path = "./models/model(loss=12.004).h5"
json_path = "./data/json/detection_config.json"

# Loading the model
detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(model_path)
detector.setJsonPath(json_path)
detector.loadModel()

# Enabling camera access
cam = cv.VideoCapture(0)

# Variables storing box coordinates
min_pt = ()
max_pt = ()


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

# Running loop for detection
while True:
    _, frame = cam.read()

    # new_t = time.time()

    # Detecting hands
    detections = detector.detectObjectsFromImage(input_image=frame, output_type="array", 
    input_type="array", minimum_percentage_probability=10, nms_treshold=0.2,
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
    
    # Displaying detected hands
    cv.imshow("Hand detector", frame)

    if cv.waitKey(1) == ord('q'):
        break

# Release resources
cam.release()
cv.destroyAllWindows()