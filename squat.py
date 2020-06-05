import argparse

import cv2
import numpy as np
from scipy import ndimage

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

# python custom.py --model=mobilenet_thin --resize=432x368


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

    cam = cv2.VideoCapture(vidlocation)

    count_of_squats = 0
    squat_pos = 0
    prev_squat_pos = 0

    i = 0
    while True:

        ret_val, image = cam.read()

        if ret_val == False:
            print("Video file not found")
            break

        i += 1
        if i < 360:
            continue

        humans = e.inference(image, resize_to_default=(
            w > 0 and h > 0), upsample_size=resize_out_ratio)

        if len(humans) < 1:
            print("No human detected")
            continue

        try:
            center_5 = (int(humans[0].body_parts[5].x * w),
                        int(humans[0].body_parts[5].y * h))       # Left shoulder
            center_11 = (int(humans[0].body_parts[11].x * w),
                            int(humans[0].body_parts[11].y * h))  # left hip
            center_12 = (int(humans[0].body_parts[12].x * w),
                            int(humans[0].body_parts[12].y * h))  # left knee
            center_13 = (int(humans[0].body_parts[13].x * w),
                            int(humans[0].body_parts[13].y * h))  # left ankle

            squat_left_angle_1 = angle_between_points(
                center_5, center_11, center_12)
            squat_left_angle_2 = angle_between_points(
                center_11, center_12, center_13)

            if squat_left_angle_2 >= 90 and squat_left_angle_2 <= 100 and squat_left_angle_1 >= 85 and squat_left_angle_1 <= 95:
                squat_pos = 1
                fontColor = (0, 255, 0)
            else:
                fontColor = (0, 0, 255)
                squat_pos = 0

            if prev_squat_pos - squat_pos == 1:
                count_of_squats += 1
            prev_squat_pos = squat_pos

            cv2.putText(image, 'Number of squats: ' + str(count_of_squats),
                        (100, 100),
                        font,
                        fontScale,
                        fontColor,
                        lineType,
                        thickness
                        )
            cv2.putText(image, 'Angle of hip joint: ' + str(round(squat_left_angle_1, 1)),
                        (100, 200),
                        font,
                        fontScale,
                        fontColor,
                        lineType,
                        thickness
                        )
            cv2.putText(image, 'Angle of knee joint: ' + str(round(squat_left_angle_2, 1)),
                        (100, 300),
                        font,
                        fontScale,
                        fontColor,
                        lineType,
                        thickness
                        )

            cv2.putText(image, 'Squat position: ' + str('Yes' if squat_pos == 1 else 'No'),
                        (100, 400),
                        font,
                        fontScale,
                        fontColor,
                        lineType,
                        thickness
                        )

            cv2.putText(image, 'Burned calories: ' + str(round(52.5 * count_of_squats, 2)),
                        (100, 500),
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
