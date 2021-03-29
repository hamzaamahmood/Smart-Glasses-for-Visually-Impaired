from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import face_recognition
import imutils
import pickle
import numpy as np
import argparse
import time
import cv2
import pytesseract 

engine = pyttsx3.init()

BUTTON_GPIO = 21
BUTTON_GPIO1 = 20
BUTTON_GPIO2 = 16
LED_GPIO = 22
LED_GPIO1 = 4
LED_GPIO2 = 3

encodingsP = "encodings.pickle"
cascade = "haarcascade_frontalface_default.xml"

print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())
detector = cv2.CascadeClassifier(cascade)

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath,configPath)


# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
#vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON_GPIO1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON_GPIO2, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(LED_GPIO, GPIO.OUT)
GPIO.setup(LED_GPIO1, GPIO.OUT)
GPIO.setup(LED_GPIO2, GPIO.OUT)

def facial_recognition():
	# grab the frame from the threaded video stream and resize it
	# to 500px (to speedup processing)
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	
	# convert the input frame from (1) BGR to grayscale (for face
	# detection) and (2) from BGR to RGB (for face recognition)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# detect faces in the grayscale frame
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

	# OpenCV returns bounding box coordinates in (x, y, w, h) order
	# but we need them in (top, right, bottom, left) order, so we
	# need to do a bit of reordering
	boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

	# compute the facial embeddings for each face bounding box
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown" #if face is not recognized, then print Unknown

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)
			
			#If someone in your dataset is identified, print their name on the screen
	#		if currentname != name:
	#			currentname = name
	#			print(currentname)
		
		# update the list of names
		names.append(name)
		else:
			print("No face detected")
			engine.say("No face detected")
	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# draw the predicted face name on the image - color is in BGR
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 225), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			.8, (0, 255, 255), 2)
		if names:
			print(name)
			engine.say(name)
		else:
			print("No face detected")
	# display the image to our screen
	engine.runAndWait()
	cv2.imshow("Facial Recognition is Running", frame)
	key = cv2.waitKey(1) & 0xFF

while True:

    if GPIO.input(BUTTON_GPIO) == GPIO.HIGH:

        print("You pushed button 1. Starting facial recognition.")
        engine.say("You pushed button 1. Starting facial recognition.")
        engine.runAndWait()
        facial_recognition()

    elif GPIO.input(BUTTON_GPIO1) == GPIO.HIGH:

        print("You pushed button 2. Starting object recognition.")
        engine.say("You pushed button 2. Starting object recognition.")
        engine.runAndWait()
        object_recognition()

    elif GPIO.input(BUTTON_GPIO2) == GPIO.HIGH:

        print("You pushed button 3. Starting text recognition.")
        engine.say("You pushed button 3. Starting text recognition.")
        engine.runAndWait()
        text_recognition()
        
GPIO.cleanup()
print("Shutting Off")

