import time
import cv2
import glob
import json
import torch
import torchaudio
import math
import sys
import re
import logging
import os, errno
import numpy as np

from videoProcess.detector import *
from videoProcess.visualization_utils import *
from videoProcess.my_utils import *

from torch.nn import functional as F
from PIL import Image


CROP_HEIGHT = 96
CROP_WIDTH = 96
logger = logging.getLogger(__name__)
# basedir_to_save = "/home2/mayipeng/myp_mutilAV/TCDTIMIT/"
# basedir = "/home3/zhangzhan/TCDTIMITprocessing/downloadTCDTIMIT/volunteers/01M/Clips/straightcam/"

def load_video(video_dir, n_videoFrame):
    video = []
    cap = cv2.VideoCapture(video_dir)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    start_frame = (n_videoFrame-1) * 30  # 120仅对应30帧视频，每次读4s
    end_frame = start_frame + 120
    # print("start_frame: ", start_frame)
    # print("end_frame: ", end_frame)
    # print("frames: ", frames)
    i = 0
    while (cap.isOpened()):
        ret, frame0 = cap.read()  # BGR
        # print("i: "+str(i))
        if ret:
            frame = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)  # RGB
            frame = cv2.resize(frame, (384, 216))
            # print(frame.shape)
            # frame = Image.fromarray(frame)  # 转换为PIL
            if (i >= start_frame) & (i < end_frame):
                video.append(frame)
        else:
            break
        i += 1
    cap.release()
    videos = np.array(video)
    videos = np.pad(videos, ((0, 120-len(videos)), (0, 0), (0, 0), (0, 0)), 'constant', constant_values=0)  # 帧数用0填充120
    data = videos  # (data: torch.Size([120, 216, 384, 3]))
    data = torch.from_numpy(data)
    return data

def load_video_onlyAudio(video_dir, n_videoFrame):
    data = torch.zeros(120, 216, 384, 3)
    return data

# def extract_opencv(filename):
#     video = []
#     video_features = []
#     time_list = []
#     cap = cv2.VideoCapture(filename)
#     frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#     # # cnt = 0
#     # first_flag = 0
#     # cropped = np.zeros((CROP_HEIGHT, CROP_WIDTH), dtype=np.uint8)
#     while (cap.isOpened()):
#         ret, frame0 = cap.read()  # BGR
#         if ret:
#             frame = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)  # RGB
#             frame = cv2.resize(frame, (384, 216))
#             # print(frame.shape)
#             frame = Image.fromarray(frame)  # 转换为PIL
#             starttime = time.time()
#             bounding_boxes, landmarks, load_time, features = detect_faces(frame)
#             endtime = time.time()
#             # print(bounding_boxes)
#             # print(features.shape)
#             time_one = endtime - starttime - load_time
#             time_list.append(time_one)
#             img_copy = show_bboxes(frame, bounding_boxes, landmarks)
#             img = cv2.cvtColor(np.asarray(img_copy), cv2.COLOR_RGB2BGR)  # BGR
#             video.append(img)
#             video_features.append(features)
#         else:
#             break
#     cap.release()
#     video_features = np.array(video_features)
#     data = video_features  # (frame, 128)
#     data = torch.from_numpy(data)
#     path_to_save = os.path.join(basedir_to_save,
#                                 filename.split('/')[-1][:-4] + '.npz')
#     if not os.path.exists(os.path.dirname(path_to_save)):
#         try:
#             os.makedirs(os.path.dirname(path_to_save))
#         except OSError as exc:
#             if exc.errno != errno.EEXIST:
#                 raise
#     np.savez(path_to_save, data=data)
#     print('total_frames: ', frames)
#
#     return data, 1

def match_dns(noisy, clean):
    """match_dns.
    Match noisy and clean DNS dataset filenames.

    :param noisy: list of the noisy filenames
    :param clean: list of the clean filenames
    """
    logger.debug("Matching noisy and clean for dns dataset")
    noisydict = {}
    extra_noisy = []
    for path, size in noisy:
        match = re.search(r'fileid_(\d+)\.wav$', path)
        if match is None:
            # maybe we are mixing some other dataset in
            extra_noisy.append((path, size))
        else:
            noisydict[match.group(1)] = (path, size)
    noisy[:] = []
    extra_clean = []
    copied = list(clean)
    clean[:] = []
    for path, size in copied:
        match = re.search(r'fileid_(\d+)\.wav$', path)
        if match is None:
            extra_clean.append((path, size))
        else:
            noisy.append(noisydict[match.group(1)])
            clean.append((path, size))
    extra_noisy.sort()
    extra_clean.sort()
    clean += extra_clean
    noisy += extra_noisy


def match_files(noisy, clean, matching="sort"):
    """match_files.
    Sort files to match noisy and clean filenames.
    :param noisy: list of the noisy filenames
    :param clean: list of the clean filenames
    :param matching: the matching function, at this point only sort is supported
    """
    if matching == "dns":
        # dns dataset filenames don't match when sorted, we have to manually match them
        match_dns(noisy, clean)
    elif matching == "sort":
        noisy.sort()
        clean.sort()
    else:
        raise ValueError(f"Invalid value for matching {matching}")

class Audioset_onlyAudio:
    def __init__(self, files=None, basedir_to_save=None, basedir=None, length=None, stride=None,
                 pad=True, with_path=False, sample_rate=None):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.length = length
        self.stride = stride or length
        self.with_path = with_path
        self.sample_rate = sample_rate
        self.basedir_to_save = basedir_to_save
        self.basedir = basedir
        for file, file_length in self.files:
            # print(file)
            if length is None:
                examples = 1
            elif file_length < length:
                examples = 1 if pad else 0
            elif pad:
                examples = int(math.ceil((file_length - self.length) / self.stride) + 1)
            else:
                examples = (file_length - self.length) // self.stride + 1
            self.num_examples.append(examples)  # 64000是一段，如果长度大于64000的话，每次向后走16000，examples是段数

    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        for (file, _), examples in zip(self.files, self.num_examples):
            if index >= examples:
                index -= examples
                continue

            num_frames = 0
            offset = 0
            n_videoFrame = 0
            if self.length is not None:
                offset = self.stride * index
                num_frames = self.length
                n_videoFrame = offset//16000 + 1
            out, sr = torchaudio.load(str(file), offset=offset, num_frames=num_frames)
            video_dir = self.basedir + str(file).split('/')[-1][:-4] + '.mp4'
            videos = load_video_onlyAudio(video_dir, n_videoFrame)
            if self.sample_rate is not None:
                if sr != self.sample_rate:
                    raise RuntimeError(f"Expected {file} to have sample rate of "
                                       f"{self.sample_rate}, but got {sr}")
            if num_frames:
                out = F.pad(out, (0, num_frames - out.shape[-1]))
            if self.with_path:
                return videos, out, file
            else:
                return videos, out

class Audioset:
    def __init__(self, files=None, basedir_to_save=None, basedir=None, length=None, stride=None,
                 pad=True, with_path=False, sample_rate=None):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.length = length
        self.stride = stride or length
        self.with_path = with_path
        self.sample_rate = sample_rate
        self.basedir_to_save = basedir_to_save
        self.basedir = basedir
        for file, file_length in self.files:
            # print(file)
            if length is None:
                examples = 1
            elif file_length < length:
                examples = 1 if pad else 0
            elif pad:
                examples = int(math.ceil((file_length - self.length) / self.stride) + 1)
            else:
                examples = (file_length - self.length) // self.stride + 1
            self.num_examples.append(examples)  # 64000是一段，如果长度大于64000的话，每次向后走16000，examples是段数

            # print(file_length, length, self.stride, examples)

    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        for (file, _), examples in zip(self.files, self.num_examples):
            # print("file: "+str(file))
            # print("index: " + str(index))
            # print("examples: ", examples)
            if index >= examples:
                index -= examples
                continue

            num_frames = 0
            offset = 0
            n_videoFrame = 0
            if self.length is not None:
                offset = self.stride * index
                num_frames = self.length
                n_videoFrame = offset//16000 + 1
            out, sr = torchaudio.load(str(file), offset=offset, num_frames=num_frames)
            # print("offset: " + str(offset))
            # print("num_frames: " + str(num_frames))
            # print("n_videoFrame: " + str(n_videoFrame))
            video_dir = self.basedir + str(file).split('/')[-1][:-4] + '.mp4'
            videos = load_video(video_dir, n_videoFrame)
            # print("audio_filename: " + str(file))
            # print("video_filename: ", self.basedir + str(file).split('/')[-1][:-4] + '.mp4')
            # print("------------------------------------------------------------------")

            if self.sample_rate is not None:
                if sr != self.sample_rate:
                    raise RuntimeError(f"Expected {file} to have sample rate of "
                                       f"{self.sample_rate}, but got {sr}")
            if num_frames:
                out = F.pad(out, (0, num_frames - out.shape[-1]))
            if self.with_path:
                return videos, out, file
            else:
                return videos, out

class NoisyCleanSet:
    def __init__(self, json_dir, basedir_to_save, basedir, matching="sort", length=None, stride=None,
                 pad=True, sample_rate=None):
        """__init__.

        :param json_dir: directory containing both clean.json and noisy.json
        :param matching: matching function for the files
        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        """
        noisy_json = os.path.join(json_dir, 'noisy.json')
        clean_json = os.path.join(json_dir, 'clean.json')
        with open(noisy_json, 'r') as f:
            noisy = json.load(f)
        with open(clean_json, 'r') as f:
            clean = json.load(f)

        match_files(noisy, clean, matching)
        kw = {'length': length, 'stride': stride, 'pad': pad, 'sample_rate': sample_rate}
        self.clean_set = Audioset(clean, basedir_to_save, basedir, **kw)
        self.noisy_set = Audioset(noisy, basedir_to_save, basedir, **kw)
        # print(len(self.clean_set))

        assert len(self.clean_set) == len(self.noisy_set)

    def __getitem__(self, index):
        return self.noisy_set[index], self.clean_set[index]

    def __len__(self):
        return len(self.noisy_set)

class NoisyCleanSet_onlyAudio:
    def __init__(self, json_dir, basedir_to_save, basedir, matching="sort", length=None, stride=None,
                 pad=True, sample_rate=None):
        """__init__.

        :param json_dir: directory containing both clean.json and noisy.json
        :param matching: matching function for the files
        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        """
        noisy_json = os.path.join(json_dir, 'noisy.json')
        clean_json = os.path.join(json_dir, 'clean.json')
        with open(noisy_json, 'r') as f:
            noisy = json.load(f)
        with open(clean_json, 'r') as f:
            clean = json.load(f)

        match_files(noisy, clean, matching)
        kw = {'length': length, 'stride': stride, 'pad': pad, 'sample_rate': sample_rate}
        self.clean_set = Audioset_onlyAudio(clean, basedir_to_save, basedir, **kw)
        self.noisy_set = Audioset_onlyAudio(noisy, basedir_to_save, basedir, **kw)
        # print(len(self.clean_set))

        assert len(self.clean_set) == len(self.noisy_set)

    def __getitem__(self, index):
        return self.noisy_set[index], self.clean_set[index]

    def __len__(self):
        return len(self.noisy_set)

if __name__=='__main__':

    video_filenames = glob.glob(os.path.join(basedir, '*.mp4'))
    audio_filenames = glob.glob(os.path.join(basedir, '*.wav'))
    total_file = len(video_filenames)
    index = 0
    segment = 4
    stride = 1
    sample_rate = 16000
    length = int(segment * sample_rate)
    stride = int(stride * sample_rate)
    kwargs = {"matching": 'sort', "sample_rate": 16000}
    pad = True
    audio_trainPath = "/home2/mayipeng/myp_mutilAV/egs/TCDTIMIT/tr/"

    print("----------------------------start------------------------------")
    tr_dataset = NoisyCleanSet(
        audio_trainPath, length=length, stride=stride, pad=pad, **kwargs)
    print("----------------------------start2------------------------------")

    for a, b in tr_dataset:
        # print(a, b)
        print(a[0].size(), b[1].size())
        print("----------------------------------------------------------")

































