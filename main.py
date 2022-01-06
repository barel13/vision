import math

import cv2 as cv
import numpy as np

# from networktables import NetworkTables

cv.namedWindow('trackbars')
cv.namedWindow('thresholds')
cv.namedWindow('noise')
min_hsv = np.array([16, 52, 45])
max_hsv = np.array([32, 255, 255])
threshold = np.array([0, 149])
noise = np.array([6, 5, 21])


def callback(_):
    pass


cv.createTrackbar('min_h', 'trackbars', min_hsv[0], 180, callback)
cv.createTrackbar('min_s', 'trackbars', min_hsv[1], 255, callback)
cv.createTrackbar('min_v', 'trackbars', min_hsv[2], 255, callback)
cv.createTrackbar('max_h', 'trackbars', max_hsv[0], 180, callback)
cv.createTrackbar('max_s', 'trackbars', max_hsv[1], 255, callback)
cv.createTrackbar('max_v', 'trackbars', max_hsv[2], 255, callback)

cv.createTrackbar('min_threshold', 'thresholds', threshold[0], 255, callback)
cv.createTrackbar('max_threshold', 'thresholds', threshold[1], 255, callback)
cv.createTrackbar('h', 'noise', noise[0], 255, callback)
cv.createTrackbar('templateWindowSize', 'noise', noise[1], 255, callback)
cv.createTrackbar('searchWindowSize', 'noise', noise[2], 255, callback)


def setup_camera(index=0):
    camera = cv.VideoCapture(index)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv.CAP_PROP_FPS, 50)
    return camera


def update_trackbars():
    min_hsv[0] = cv.getTrackbarPos('min_h', 'trackbars')
    min_hsv[1] = cv.getTrackbarPos('min_s', 'trackbars')
    min_hsv[2] = cv.getTrackbarPos('min_v', 'trackbars')

    max_hsv[0] = cv.getTrackbarPos('max_h', 'trackbars')
    max_hsv[1] = cv.getTrackbarPos('max_s', 'trackbars')
    max_hsv[2] = cv.getTrackbarPos('max_v', 'trackbars')

    threshold[0] = cv.getTrackbarPos('min_threshold', 'thresholds')
    threshold[1] = cv.getTrackbarPos('max_threshold', 'thresholds')

    noise[0] = cv.getTrackbarPos('h', 'noise')
    noise[1] = cv.getTrackbarPos('templateWindowSize', 'noise')
    noise[2] = cv.getTrackbarPos('searchWindowSize', 'noise')


def apply_filters(frame, min_hsv_const, max_hsv_const, kernel=5):
    hsv_mask = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    hsv_mask = cv.inRange(hsv_mask, min_hsv_const, max_hsv_const)

    frame = cv.GaussianBlur(frame, (kernel, kernel), 0)
    frame = cv.medianBlur(frame, kernel)

    hsv_mask = cv.erode(hsv_mask, (kernel, kernel), iterations=3)
    hsv_mask = cv.dilate(hsv_mask, (kernel, kernel), iterations=3)
    return cv.bitwise_and(frame, frame, mask=hsv_mask)


def detect_contours(frame):
    edge_frame = cv.Canny(frame, threshold[0], threshold[1])
    edge_frame = cv.threshold(edge_frame, 127, 255, cv.THRESH_BINARY)[1]
    # edge_frame = cv.morphologyEx(edge_frame, cv.MORPH_OPEN, (1, 1))
    contours, hierarchy = cv.findContours(edge_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    return edge_frame, contours


def get_pitch(vertical_focal_length, y):
    return -math.degrees(math.atan(y / vertical_focal_length))


def get_yaw(horizontal_focal_length, x):
    return math.degrees(math.atan(x / horizontal_focal_length))


def main():
    camera = setup_camera()
    # NetworkTables.initialize(server='10.59.87.2')
    # table = NetworkTables.getTable('vision')
    diagonal_aspect = math.hypot(640, 480)
    middle_fov = math.radians(75 / 2)
    hor_focal = 640 / (2 * math.tan(middle_fov * (640 / diagonal_aspect)))
    ver_focal = 480 / (2 * math.tan(middle_fov * (480 / diagonal_aspect)))

    while cv.waitKey(1) & 0xFF not in [27, ord('q')]:
        has_frame, frame = camera.read()
        if not has_frame:
            print("You are a loser! Why? because...")
            continue
        update_trackbars()

        lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
        lab_planes = [*cv.split(lab)]

        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv.merge(lab_planes)
        equ = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
        equ = cv.GaussianBlur(equ, (5, 5), 0)
        equ = cv.medianBlur(equ, 5)
        equ = cv.fastNlMeansDenoisingColored(equ, None, *noise.tolist())
        frame_new = np.hstack((frame, equ))



        # frame_after_hsv = apply_filters(frame, min_hsv, max_hsv, 7)
        # gray_frame = cv.cvtColor(frame_after_hsv, cv.COLOR_BGR2GRAY)
        # edge_frame, contours = detect_contours(gray_frame)

        # contours = sorted(contours, key=cv.contourArea, reverse=True)
        # correct_contours = []
        # for i, contour in enumerate(contours):
        #     area = cv.contourArea(contour)
        #     if area > 50:
        #         x, y, w, h = cv.boundingRect(contour)
        #         hull = cv.convexHull(contour)
        #         hullArea = cv.contourArea(hull)
        #         solidity = area / float(hullArea)
        #         extent = area / float(w * h)
        #         if 0 <= solidity <= 0.25 and 0 <= extent <= 0.25:
        #             correct_contours.append(contour)
        #             M = cv.moments(contour)
        #             cX = int(M["m10"] / (M["m00"] + 1e-5))
        #             cY = int(M["m01"] / (M["m00"] + 1e-5))
        #             pitch = get_pitch(ver_focal, cY)
        #             yaw = get_yaw(hor_focal, cX)
        #             # table.putNumber(f"yaw_{i}", yaw)
        #             # table.putNumber(f"pitch_{i}", pitch)
        #
        # cv.drawContours(frame, correct_contours, -1, (0, 255, 0), thickness=4)
        #
        cv.imshow("Frame", frame)
        cv.imshow("frame new", frame_new)

        # cv.imshow("hsv", frame_after_hsv)
        # cv.imshow("edge", edge_frame)

    camera.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
