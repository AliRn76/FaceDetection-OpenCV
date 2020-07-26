import cv2
import dlib
import sys


# set up the 68 point facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

video_capture = cv2.VideoCapture(0)

# bring in the input image
# img = cv2.imread('pic.jpg', 1)
number = 0
while True:
    ret, img = video_capture.read()
    # convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect faces in the image
    faces_in_image = detector(img_gray, 0)



    # loop through each face in image
    for face in faces_in_image:

        # assign the facial landmarks
        landmarks = predictor(img_gray, face)

        # unpack the 68 landmark coordinates from the dlib object into a list
        landmarks_list = []
        for i in range(0, landmarks.num_parts):
            landmarks_list.append((landmarks.part(i).x, landmarks.part(i).y))

        # for each landmark, plot and write number
        for landmark_num, xy in enumerate(landmarks_list, start=1):
            cv2.circle(img, (xy[0], xy[1]), 12, (168, 0, 20), -1)
            cv2.putText(img, str(landmark_num),(xy[0]-7,xy[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255), 1)


    # visualise the image with landmarks
    cv2.imshow('Video', img)
    # number = number+1
    # print(number)
    # cv2.imwrite('img'+ str(number) +'.jpg', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
# cv2.waitKey(0)
# cv2.destroyAllWindows()







#
# cascPath = sys.argv[1]
# faceCascade = cv2.CascadeClassifier(cascPath)
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
