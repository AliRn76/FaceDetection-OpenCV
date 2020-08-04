import cv2
import dlib
import time
import numpy as np


# set up the 68 point facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

video_capture = cv2.VideoCapture(0)

time.sleep(3)
background = 0
for i in range(30):
    ret, background = video_capture.read()

background = cv2.flip(background, 1)

while True:
    ret, img = video_capture.read()
    img = cv2.flip(img, 1)
    # convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect faces in the image
    faces_in_image = detector(img_gray, 0)

    # loop through each face in image
    for face in faces_in_image:
        points = []
        points1 = []
        points2 = []

        # assign the facial landmarks
        landmarks = predictor(img_gray, face)

        # unpack the 68 landmark coordinates from the dlib object into a list
        landmarks_list = []
        for i in range(0, landmarks.num_parts):
            landmarks_list.append((landmarks.part(i).x, landmarks.part(i).y))

        # for each landmark, plot and write number
        for landmark_num, xy in enumerate(landmarks_list, start=1):
          # We just want the border of face
            if landmark_num < 18:
                points1.append((xy[0], xy[1]))
            elif landmark_num <= 27:
                points2.append((xy[0], xy[1]))

        points2.reverse()
        points = points1 + points2
        nds = np.array(points)
        nds = np.int32([nds])

        mask = np.zeros(img.shape, dtype=np.uint8)
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
        cv2.fillPoly(mask, nds, ignore_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        # I Don't know why it can't merge them well :\
        dst = cv2.addWeighted(background, 1, masked_image, 1, 0)

    cv2.imshow('Video', dst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()








# # cascPath = sys.argv[1]
# # faceCascade = cv2.CascadeClassifier(cascPath)
# faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# video_capture = cv2.VideoCapture(0)
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#
# i = 0
# while True:
#     # Capture frame-by-frame
#     ret, frame = video_capture.read()
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     faces = faceCascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30),
#         flags=cv2.CASCADE_SCALE_IMAGE
#
#     )
#
#     # Draw a rectangle around the faces
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#
#     # Display the resulting frame
#     cv2.imshow('Video', frame)
#     i = i+1
#     # cv2.imwrite('img'+ str(i) +'.jpg', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything is done, release the capture
# video_capture.release()
# # cv2.destroyAllWindows()
