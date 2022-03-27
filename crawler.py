import time
from queue import Queue
from threading import Thread

import cv2
import requests
import m3u8
import os

from absl import app, flags, logging
from absl.flags import FLAGS

import detect_video
from Mosse_Tracker.TrackerManager import TrackerType, Tracker
from crash_detect import detect
from yolov3_tf2.models import YoloV3, YoloV3Tiny



flags.DEFINE_string('cameraNum', 'b9beaf17-934f-4f30-8c3a-b3e4c4937f3c', 'camera number in web')

headers = {    'Accept': '*/*',
               'Accept-Encoding': 'gzip, deflate, br',
                'Accept-Language': 'zh-CN,zh;q=0.9',
                'Cache-Control': 'max-age=0',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36',
                'Connection': 'keep-alive',
                'Referer': 'https://weixin.hngscloud.com/'}

tracker_type = TrackerType.MOSSE

def get_ts(url):
    try:
        response = requests.get(url, verify=False)
        response.raise_for_status()
        response.encoding = 'utf-8'
        return response.content
    except Exception as err:
        print(err)
        return b''


def save_ts(url, index):
    filename = os.path.join('./video', str(index).zfill(5) + '.ts')
    with open(filename, 'wb') as f:
        ts = get_ts(url)
        f.write(ts)
    print(filename + ' is ok!')

def main(_argv):
    apiUrl = "https://weixin.hngscloud.com/"
    res = requests.get(url=apiUrl + 'camera/playUrl?cameraNUm=' + FLAGS.cameraNum + '&videoType=2&videoRate=0',
                       headers=headers)
    playUrl = ""
    if (res.status_code == 200):
        playUrl = res.json()['data']['playUrl']
    else:
        print("请求源地址出错,错误信息为:" + res.json())

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    m3u8Url = m3u8.load(uri=playUrl, headers=headers)

    urlPrefix = m3u8Url.playlists[0].uri.rsplit('/', 1)[0] + "/"

    frame_queue = Queue()

    thread_detect = Thread(target=detect, args=(frame_queue, yolo, class_names))
    thread_detect.start()

    while(True):

        playlist = m3u8.load(uri=m3u8Url.playlists[0].uri, headers=headers)

        # # 线程池，引入index可以防止合成时视频发生乱序
        # with ThreadPoolExecutor(max_workers=10) as pool:
        #     for index, seg in enumerate(playlist.segments):
        #         pool.submit(save_ts, urlPrefix+seg.uri, index)
        #
        # files = glob.glob(os.path.join('./video', '*.ts'))
        # for file in files:
        #     with open(file, 'rb') as fr, open('./video_de/monitor' + str(i) + '.mp4', 'ab') as fw:
        #         content = fr.read()
        #         fw.write(content)
        #     print(file + ' is ok!')


        for index, seg in enumerate(playlist.segments):
            vid = cv2.VideoCapture(urlPrefix+seg.uri)

            while True:
                ret, frame = vid.read()
                if ret:
                    frame_queue.put(frame)
                else:
                    break
        time.sleep(15)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass