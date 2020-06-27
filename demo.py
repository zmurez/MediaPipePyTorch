import numpy as np
import torch
import cv2

from blazebase import resize_pad, denormalize_detections
from blazeface import BlazeFace
from blazepalm import BlazePalm
from blazeface_landmark import BlazeFaceLandmark
from blazehand_landmark import BlazeHandLandmark

from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)



face_detector = BlazeFace().to(gpu)
face_detector.load_weights("blazeface.pth")
face_detector.load_anchors("anchors_face.npy")

palm_detector = BlazePalm().to(gpu)
palm_detector.load_weights("blazepalm.pth")
palm_detector.load_anchors("anchors_palm.npy")

hand_regressor = BlazeHandLandmark().to(gpu)
hand_regressor.load_weights("blazehand_landmark.pth")

face_regressor = BlazeFaceLandmark().to(gpu)
face_regressor.load_weights("blazeface_landmark.pth")


WINDOW='test'
cv2.namedWindow(WINDOW)
capture = cv2.VideoCapture(0)

if capture.isOpened():
    hasFrame, frame = capture.read()
    frame_ct = 0
else:
    hasFrame = False

while hasFrame:
    frame_ct +=1

    frame = np.ascontiguousarray(frame[:,::-1,::-1])

    img1, img2, scale, pad = resize_pad(frame)

    normalized_face_detections = face_detector.predict_on_image(img2)
    normalized_palm_detections = palm_detector.predict_on_image(img1)

    face_detections = denormalize_detections(normalized_face_detections, scale, pad)
    palm_detections = denormalize_detections(normalized_palm_detections, scale, pad)


    xc, yc, scale, theta = face_detector.detection2roi(face_detections)
    img, affine, box = face_regressor.extract_roi(frame, xc, yc, theta, scale)
    flags, landmarks = face_regressor(img)


    xc, yc, scale, theta = palm_detector.detection2roi(palm_detections)
    img, affine2, box2 = hand_regressor.extract_roi(frame, xc, yc, theta, scale)
    flags2, handed2, landmarks2 = hand_regressor(img)

    

    for i in range(len(flags)):
        landmark, flag, M = landmarks[i], flags[i], affine[i]
        if flag>.5:
            landmark = landmark[:,:2]*192
            landmark = (M[:,:2] @ landmark.T + M[:,2:]).T
            draw_landmarks(frame, landmark, size=1)


    for i in range(len(flags2)):
        landmark, flag, M = landmarks2[i], flags2[i], affine2[i]
        if flag>.5:
            landmark = landmark[:,:2]*256
            landmark = (M[:,:2] @ landmark.T + M[:,2:]).T
            draw_landmarks(frame, landmark, HAND_CONNECTIONS, size=2)

    draw_roi(frame, box)
    draw_roi(frame, box2)
    draw_detections(frame, face_detections)
    draw_detections(frame, palm_detections)

    cv2.imshow(WINDOW, frame[:,:,::-1])
    # cv2.imwrite('sample/%04d.jpg'%frame_ct, frame[:,:,::-1])

    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
