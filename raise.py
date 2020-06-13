import argparse

import cv2
import numpy as np
from scipy import ndimage

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def angle_between_points(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 1.5
lineType = 2
thickness = 5


if __name__ == '__main__':
    # Initial arguments

    resize = '432x368'  # Recommended 432x368 or 656x368 or 1312x736
    resize_out_ratio = 4.0
    model = 'mobilenet_thin'
    vidlocation = 'squat.mp4'
    tensorrt = "False"

    w, h = model_wh(resize)
    print(w, h)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(model), target_size=(
            w, h), trt_bool=str2bool(tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(model), target_size=(
            432, 368), trt_bool=str2bool(tensorrt))

    cam = cv2.VideoCapture(0)

    count_of_raises = 0
    raise_pos = 0
    prev_raise_pos = 0

    i = 0
    while True:

        ret_val, image = cam.read()

        if ret_val == False:
            print("Video file not found")
            break

        humans = e.inference(image, resize_to_default=(
            w > 0 and h > 0), upsample_size=resize_out_ratio)

        if len(humans) < 1:
            print("No human detected")
            continue

        try:
            center_1 = (int(humans[0].body_parts[1].x * w),
                        int(humans[0].body_parts[1].y * h))  # sternum
            center_5 = (int(humans[0].body_parts[5].x * w),
                        int(humans[0].body_parts[5].y * h))  # left shoulder
            center_6 = (int(humans[0].body_parts[6].x * w),
                        int(humans[0].body_parts[6].y * h))  # left elbow

            raise_angle = angle_between_points(center_1, center_5, center_6)
        
            if raise_angle >= 150:
                raise_pos = 1
                fontColor = (0,255,0)
            else:
                fontColor = (0,0,255)
                raise_pos = 0
            
            if prev_raise_pos - raise_pos == 1:
                count_of_raises +=1
            prev_raise_pos = raise_pos

            cv2.putText(image, 'Number of raises: ' + str(count_of_raises),
                        (0, 0),
                        font,
                        fontScale,
                        fontColor,
                        lineType,
                        thickness
                        )
            cv2.putText(image, 'Angle of shoulder: ' + str(round(raise_angle, 1)),
                        # (100, 200),
                        center_5,
                        font,
                        fontScale,
                        fontColor,
                        lineType,
                        thickness
                        )
        except:
            print("Incorrect camera dimensions")

        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        cv2.imshow('tf-pose-estimation result',
                   cv2.resize(image, (0, 0), fx=0.5, fy=0.5))

        if cv2.waitKey(1) == 'q':
            break

    cam.release()
    cv2.destroyAllWindows()
