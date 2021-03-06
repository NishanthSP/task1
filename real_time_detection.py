import cv2

# Initialize the web camera with specifications
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)


# Lists of objects that can be detected by this model
classNames = []
classFile = 'labels.txt'

with open(classFile, 'rt') as f:
	classNames = f.read().rstrip().split()


# Linking the Single-Shot multibox Detection network
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'


# Creating dnn detection model with certain specifications
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


# Detect the objects in real time and the edit and display the corresponding objects name.
while True:
	success, img = cap.read()


	if success is True:
		classIds, conf, bbox = net.detect(img, confThreshold=0.45)

		if len(classIds) != 0:
			for classId, confidence, box in zip(classIds.flatten(), conf.flatten(), bbox):
				cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
				cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
				cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

		cv2.imshow("Output", img)

# Press 'q' to terminate the web camera

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break

cap.release()
cv2.destroyAllWindows()
