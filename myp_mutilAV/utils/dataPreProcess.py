import glob
import os

basedir_to_save = "/home2/mayipeng/myp_mutilAV/TCDTIMIT/"
basedir = "/home3/zhangzhan/TCDTIMITprocessing/downloadTCDTIMIT/volunteers/"
video_filenames = glob.glob(os.path.join(basedir, '*', '*', '*', '*.mp4'))
audio_filenames = glob.glob(os.path.join(basedir, '*', '*', '*', '*.wav'))
total_file = len(video_filenames)
index = 0

print("----------------------------start------------------------------")
while index < total_file:
    audio_filename = audio_filenames[index]
    print("audio_filename: " + audio_filename)



























