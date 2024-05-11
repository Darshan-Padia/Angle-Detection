# program to capture single image from webcam in python

# importing OpenCV library
import cv2
import finall
import sys
import time

# print(sys.argv[0])
name = sys.argv[1]
# name = "kin"

# initialize the camera
# If you have multiple camera connected with
# current device, assign a value in cam_port
# variable according to that
cap = cv2.VideoCapture(0)


while cap.isOpened():
	ret, frame = cap.read()
	
	
	# Flip on horizontal
	image = cv2.flip(frame, 1)
	
	

		
	cv2.imshow('Hand Tracking', image)

	if cv2.waitKey(10) & 0xFF == ord('q'):
		cv2.imwrite(f"/home/darshan/college/fing_rom_app/ImagesAttendance/{name}.jpg",image)
		break

cap.release()
cv2.destroyAllWindows()
