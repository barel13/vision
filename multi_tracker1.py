from imutils.video import VideoStream
import multiprocessing
import imutils
import argparse
import cv2
import time

from stranger_danger import hsv

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
                help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
                help="OpenCV object tracker type")
args = vars(ap.parse_args())


OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

redUpper = (9, 255, 255)
redLower = (0, 121, 0)

# initialize OpenCV's special multi-object tracker
trackers = cv2.MultiTracker_create()

if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])
# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    # check to see if we have reached the end of the stream
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    mask = cv2.inRange(hsv, redLower, redUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        # c = max(cnts, key=cv2.contourArea)
        for c in cnts:
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # only proceed if the radius meets a minimum size

            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
    cv2.imshow("multi", frame)
    if frame is None:
        break

    # resize the frame (so we can process it faster)
    frame = imutils.resize(frame, width=600)
    (success, boxes) = trackers.update(frame)
    #
    # # loop over the bounding boxes and draw then on the frame
    # for box in boxes:
    #     (x, y, w, h) = [int(v) for v in box]
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     cv2.imshow("Frame", frame)
    #     key = cv2.waitKey(1) & 0xFF
    #
    #     # if the 's' key is selected, we are going to "select" a bounding
    #     # box to track
    #     if key == ord("s"):
    #         # select the bounding box of the object we want to track (make
    #         # sure you press ENTER or SPACE after selecting the ROI)
    #         box = cv2.selectROI("Frame", frame, fromCenter=False,
    #                             showCrosshair=True)
    #         # create a new object tracker for the bounding box and add it
    #         # to our multi-object tracker
    #         tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
    #         trackers.add(tracker, frame, box)
    #         if key == ord("q"):
    #             break

        # # if we are using a webcam, release the pointer
        # if not args.get("video", False):
        #     vs.stop()
        # # otherwise, release the file pointer
        #
        # else:
        #     vs.release()
        # # close all windows
        # cv2.destroyAllWindows()
