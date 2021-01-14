from detector import *
from visualization_utils import *
from PIL import Image
import time

import os, errno
import cv2
import glob
import numpy as np
from my_utils import *


CROP_HEIGHT = 96
CROP_WIDTH = 96

def extract_opencv(filename):
    miss = 0
    sum_time = 0.0
    video = []
    video_features = []
    time_list = []
    cap = cv2.VideoCapture(filename)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # # cnt = 0
    # first_flag = 0
    # cropped = np.zeros((CROP_HEIGHT, CROP_WIDTH), dtype=np.uint8)
    while (cap.isOpened()):
        ret, frame0 = cap.read()  # BGR
        if ret:
            frame = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)  # RGB
            frame = cv2.resize(frame, (384, 216))
            # print(frame.shape)
            frame = Image.fromarray(frame)  # 转换为PIL
            starttime = time.time()
            bounding_boxes, landmarks, load_time, features = detect_faces(frame)
            endtime = time.time()
            # print(bounding_boxes)
            # print(features.shape)
            time_one = endtime - starttime - load_time
            time_list.append(time_one)
            img_copy = show_bboxes(frame, bounding_boxes, landmarks)
            img = cv2.cvtColor(np.asarray(img_copy), cv2.COLOR_RGB2BGR)  # BGR
            video.append(img)
            video_features.append(features)
            # if (bounding_boxes[0][-1]<0.1):
            #     miss = miss + 1
            if (bounding_boxes == []):
                miss = miss + 1
        else:
            break
    cap.release()
    for t in time_list:
        sum_time = sum_time + t
    average_time = sum_time / len(time_list)
    # print("average_time: " + str(1 / average_time) + " FPS")
    # print(np.shape(video))
    video = np.array(video)
    video_features = np.array(video_features)
    # print(video.shape)
    # change BGR to RGB, video(frame,width,height,channel)
    # video = video[...,::-1]
    # data = video.reshape((video.shape[0], 3, video.shape[1], video.shape[2]))
    data = video_features  # (frame, 128)
    path_to_save = os.path.join(basedir_to_save,
                                filename.split('/')[-4],
                                filename.split('/')[-3],
                                filename.split('/')[-2],
                                filename.split('/')[-1][:-4] + '.npz')
    if not os.path.exists(os.path.dirname(path_to_save)):
        try:
            os.makedirs(os.path.dirname(path_to_save))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    np.savez(path_to_save, data=data)
    print('miss: ', miss, 'total: ', frames, 'bad rate: ', miss / frames, 'FPS: ', 1 / average_time, ' FPS')
    print("-----------------------------------------------------------")

    if miss / frames > 0.1:
        with open("/home2/mayipeng/mtcnn/lrs2/bad_record.txt", 'a+') as log:
            log.write(filename + '\n')
        return 0
    else:
        return 1

basedir_to_save = "/home2/mayipeng/mtcnn/lrs2/"
# basedir = "/home3/zhangzhan/TCDTIMITprocessing/downloadTCDTIMIT/volunteers/"
basedir = "/home3/zhangzhan/lrs2/mvlrs_v1/pretrain/"
# video = extract_opencv("/home3/zhangzhan/lrw/lipread_mp4/ABOUT/test/ABOUT_00001.mp4")
# filenames = glob.glob(os.path.join(basedir, '*', '*', '*', '*.mp4'))
filenames = glob.glob(os.path.join(basedir, '*', '*.mp4'))
total_file = len(filenames)
index = 0
good_nums = 0
good_flag = 1

print("----------------------------start------------------------------")
while index < total_file:
    # print("index: " + str(index))
    filename = filenames[index]
    print("filename: " + filename)
    good_flag = extract_opencv(filename)
    if good_flag:
        good_nums = good_nums + 1
    index = index + 1
    if index >= total_file:
        break

print("good_per: ", good_nums / total_file)


# 参数量：6510+99636+387936=494082
# 模型：28+393+1480=1.9 MB



