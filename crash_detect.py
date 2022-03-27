import logging
import os
import time
from queue import Queue
from threading import Thread

import cv2
import tensorflow as tf
from absl import app

from VIF.vif import VIF
from absl.flags import FLAGS

from Mosse_Tracker.TrackerManager import TrackerType, Tracker
from Mosse_Tracker.utils import draw_str
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.models import YoloV3Tiny, YoloV3
import detect_video

pi=22/7
vif = VIF()
tracker_type = TrackerType.MOSSE

def predict(frames_RGB,trackers):
    gray_frames = []
    for frame in frames_RGB:
        gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    no_crash = 0
    crash = 0

    for tracker in trackers:
        tracker_frames,width,height,xmin,xmax,ymin,ymax = tracker.getFramesOfTracking(gray_frames)

        if tracker_frames == None:
            continue

        if xmax - xmin < 50: #50
            continue
        if ymax - ymin <= 28: #35
            continue

        if (ymax- ymin) / (xmax - xmin) <0.35: #0.35
            continue

        feature_vec = vif.process(tracker_frames)
        result = vif.clf.predict(feature_vec.reshape(1, 304))
        if result[0] == 0.0:
            no_crash += 1
        else:
            crash += 1
            tracker.saveTracking(frames_RGB)

# 两两查询距离,并进行碰撞预测
def process(trackers,frames):

    new_trackers = trackers
    for i in range(len(new_trackers)):
        for j in range(i+1,len(trackers)):
            if i == j:
                continue
            tracker_A = trackers[i]
            tracker_B = trackers[j]

            if  checkDistance(frames,tracker_A,tracker_B,16) or checkDistance(frames,tracker_A,tracker_B,19) or checkDistance(frames,tracker_A,tracker_B,22) or checkDistance(frames,tracker_A,tracker_B,25) or checkDistance(frames,tracker_A,tracker_B,28):
                predict(frames, [tracker_B,tracker_A])

# 距离检测
def checkDistance(frames,tracker_A,tracker_B,frame_no):
    # 如果不超速,就直接返回false
    if not tracker_A.isAboveSpeedLimit(frame_no-10,frame_no) and not tracker_B.isAboveSpeedLimit(frame_no-10,frame_no) :
        return False

    xa, ya = tracker_A.estimationFutureCenter[frame_no]
    xb, yb = tracker_B.estimationFutureCenter[frame_no]
    r = pow(pow(xa - xb, 2) + pow(ya - yb, 2), 0.5)
    tracker_A_area = 0.5 * tracker_A.width * tracker_A.height
    tracker_B_area = 0.5 * tracker_B.width * tracker_B.height

    if tracker_type == TrackerType.MOSSE:
        xa_actual,ya_actual = tracker_A.tracker.centers[frame_no]
        xb_actual,yb_actual = tracker_B.tracker.centers[frame_no]
    else:
        xa_actual,ya_actual = tracker_A.get_position(tracker_A.history[frame_no])
        xb_actual,yb_actual = tracker_B.get_position(tracker_B.history[frame_no])
    difference_trackerA_actual_to_estimate = pow(pow(xa_actual - xa, 2) + pow(ya_actual - ya, 2), 0.5)
    difference_trackerB_actual_to_estimate = pow(pow(xb_actual - xb, 2) + pow(yb_actual - yb, 2), 0.5)
    max_difference = max(difference_trackerA_actual_to_estimate,difference_trackerB_actual_to_estimate)
    if r == 0:
        return True

    if r < 40 and max_difference/r > 0.5:
        return True
    return False

def show(frame_queue):
    #按一定速率进行输出
    while True:
        if frame_queue.qsize() < 1:
            time.sleep(2)
            continue

        time.sleep(0.03)
        frame = frame_queue.get()
        cv2.imshow("output", frame)
        if cv2.waitKey(1) == ord('q'):
            break


def detect(frame_queue, yolo, class_names):

    show_frame_queue = Queue()
    trackers = []
    animal_trackers = []
    last_30_frames = []
    trackerId = 0

    # 下次进行yolo算法的帧数
    frameCount = 0
    fps = 30

    thread_show = Thread(target=show, args=(show_frame_queue,))
    thread_show.start()

    while True:
        if frame_queue.qsize() < 1:
            time.sleep(2)
            continue

        frame = frame_queue.get()

        frameVif = cv2.resize(frame, (FLAGS.size, FLAGS.size), interpolation=cv2.INTER_AREA)
        new_frame = frameVif.copy()

        # 对前30帧进行碰撞判断
        if frameCount > 0 and frameCount % fps == 0 :
            process(trackers, last_30_frames)

        # Call Yolo
        if frameCount % fps == 0 or frameCount == 0:

            frameYolo = tf.expand_dims(frame, 0)
            frameYolo = transform_images(frameYolo, FLAGS.size)

            trackers = []
            last_30_frames = []

            boxes, scores, classes, nums = yolo.predict(frameYolo)
            for index, box in enumerate(boxes.tolist()[0]):
                if index >= nums:
                    break

                label = class_names[int(classes[0][index])]

                xmin = int(box[0] * FLAGS.size)
                ymin = int(box[1] * FLAGS.size)
                xmax = int(box[2] * FLAGS.size)
                ymax = int(box[3] * FLAGS.size)

                trackerId += 1

                if xmax < FLAGS.size and ymax < FLAGS.size:
                    tr = Tracker(frameVif, (xmin, ymin, xmax, ymax), FLAGS.size, FLAGS.size,
                                 trackerId, tracker_type, label)
                    trackers.append(tr)
                elif xmax < FLAGS.size and ymax >= FLAGS.size:
                    tr = Tracker(frameVif, (xmin, ymin, xmax, FLAGS.size - 1), FLAGS.size,
                                 FLAGS.size, trackerId, tracker_type, label)
                    trackers.append(tr)
                elif xmax >= FLAGS.size and ymax < FLAGS.size:
                    tr = Tracker(frameVif, (xmin, ymin, FLAGS.size - 1, ymax), FLAGS.size, FLAGS.size,
                                 trackerId, tracker_type, label)
                    trackers.append(tr)
                else:
                    tr = Tracker(frameVif, (xmin, ymin, FLAGS.size - 1, FLAGS.size - 1), FLAGS.size,
                                 FLAGS.size, trackerId, tracker_type, label)
                    trackers.append(tr)

        else:

            for i, tracker in enumerate(trackers):
                left, top, right, bottom = tracker.update(frameVif)
                redian = tracker.getCarAngle() * (pi / 180)
                redian = 0

                left_future, top_future, right_future, bottom_future = tracker.futureFramePosition()

                if left > 0 and top > 0 and right < FLAGS.size and bottom < FLAGS.size:
                    if tracker.isAboveSpeedLimit():
                        cv2.rectangle(frameVif, (int(left), int(top)), (int(right), int(bottom)),
                                      (0, 0, 255))  # B G R

                    elif tracker.isUnderSpeedLimit():
                        cv2.rectangle(frameVif, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0))

                    else:
                        cv2.rectangle(frameVif, (int(left), int(top)), (int(right), int(bottom)), (225, 225, 225))

                    draw_str(frameVif, (int(left), int(bottom) + 16), tracker.label + ' Avg Speed: %.2f' % tracker.getAvgSpeed())

                if left_future > 0 and top_future > 0 and right_future < FLAGS.size and bottom_future < FLAGS.size:
                    cv2.rectangle(frameVif, (int(left_future), int(top_future)),
                                  (int(right_future), int(bottom_future)), (0, 255, 0))
        frameCount += 1
        show_frame_queue.put(frameVif)
        last_30_frames.append(new_frame)
        if cv2.waitKey(1) == ord('q'):
            break

def main(_argv):
    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    frame_queue = Queue()

    thread_detect = Thread(target=detect, args=(frame_queue, yolo, class_names))
    thread_detect.start()

    vid = cv2.VideoCapture("videos/1534.mp4")

    while True:
        ret, frame = vid.read()
        if ret:
            frame_queue.put(frame)
        else:
            break

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass




