import cv2


# Linking the Single-Shot multibox Detection network
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'


# Creating dnn detection model with certain specifications
model = cv2.dnn_DetectionModel(frozen_model, config_file)

model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)


# Lists of objects that can be detected by this model
classLabels = []
file_name = 'labels.txt'


# Get the list of the objects into a list by reading the txt file
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip().split()


# Read the image
img = cv2.imread("man-car.jpg")


# Get the information (like type of objects, coordinates of bounding boxes) about the image from the model
ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)


# From the information obtained above edit the image with bounding boxes and corresponding texts.
if len(ClassIndex) > 0:
    for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
        cv2.rectangle(img, boxes, (255, 0, 0), 2)
        cv2.putText(img, text=classLabels[ClassInd-1], org=(boxes[0]+10, boxes[1]+30), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(0, 0, 0), thickness=3)

# Show the edited image
img = cv2.resize(img, (600, 600))
cv2.imshow("Output", img)

cv2.waitKey(0)
