import numpy as np
import torch
import cv2

from blazebase import resize_pad, denormalize_detections
from blazepose import BlazePose
from blazepose_landmark import BlazePoseLandmark

from visualization import draw_detections, draw_landmarks, draw_roi

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)



pose_detector = BlazePose().to(gpu)
pose_detector.load_weights("blazepose.pth")
pose_detector.load_anchors("anchors_pose.npy")

pose_regressor = BlazePoseLandmark().to(gpu)
pose_regressor.load_weights("blazepose_landmark.pth")


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

    normalized_pose_detections = pose_detector.predict_on_image(img2)
    pose_detections = denormalize_detections(normalized_pose_detections, scale, pad)

    xc, yc, scale, theta = pose_detector.detection2roi(pose_detections)
    img, affine, box = pose_regressor.extract_roi(frame, xc, yc, theta, scale)
    flags, landmarks, mask = pose_regressor(img)


    draw_detections(frame, pose_detections)
    draw_roi(frame, box)

    for i in range(len(flags)):
        landmark, flag, M = landmarks[i], flags[i], affine[i]
        print(flag)
        if flag>.5:
            landmark = landmark[:,:2]*256
            landmark = (M[:,:2] @ landmark.T + M[:,2:]).T
            draw_landmarks(frame, landmark, size=1)


    cv2.imshow(WINDOW, frame[:,:,::-1])
    # cv2.imwrite('sample/%04d.jpg'%frame_ct, frame[:,:,::-1])

    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
