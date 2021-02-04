# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import json
from pathlib import Path
import glob
import random
import os
import sys

import torchaudio



def find_audio_files(path, exts=[".wav"], progress=True):
    audio_files = []
    for root, folders, files in os.walk(path, followlinks=True):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() in exts:  # file.suffix.lower()获取文件后缀
                audio_files.append(str(file.resolve()))
    meta = []
    for idx, file in enumerate(audio_files):
        siginfo, _ = torchaudio.info(file)
        length = siginfo.length // siginfo.channels
        meta.append((file, length))
        if progress:
            print(format((1 + idx) / len(audio_files), " 3.1%"), end='\r', file=sys.stderr)
    meta.sort()
    return meta

# if __name__ == "__main__":
#     meta = []
#     for path in sys.argv[1:]:
#         meta += find_audio_files(path)
#     json.dump(meta, sys.stdout, indent=4)

def gen_meta(path):
    meta = []
    siginfo, _ = torchaudio.info(path)
    length = siginfo.length // siginfo.channels
    meta.append((path, length))
    meta.sort()
    return meta

if __name__ == "__main__":

    basedir = "/home2/mayipeng/myp_mutilAV/data/audio/u/drspeech/data/TCDTIMIT/Noisy_TCDTIMIT/"
    cleanBasedir = "/home2/mayipeng/myp_mutilAV/data/audio/Clean/volunteers/"
    filePaths = glob.glob(basedir + '*/'+ '*'+'/volunteers/' +'*/'+'*/'+'*.wav')

    fileOutPath_tr_noisy = "/home2/mayipeng/myp_mutilAV/egs/TCDTIMIT/tr/noisy.json"
    fileOutPath_tr_clean = "/home2/mayipeng/myp_mutilAV/egs/TCDTIMIT/tr/clean.json"

    fileOutPath_tt_noisy = "/home2/mayipeng/myp_mutilAV/egs/TCDTIMIT/tt/noisy.json"
    fileOutPath_tt_clean = "/home2/mayipeng/myp_mutilAV/egs/TCDTIMIT/tt/clean.json"

    index = 0
    total_file = len(filePaths)
    random.shuffle(filePaths)

    filePaths = filePaths[0:total_file//10]

    filePaths_tr = filePaths[0:total_file//100*7]
    filePaths_tt = filePaths[total_file//100*7:total_file//10]

    # print(len(filePaths))
    meta_clean_tr = []
    meta_dirty_tr = []

    meta_clean_tt = []
    meta_dirty_tt = []

    print("--------------------tr_start--------------------")
    for filePath_tr in filePaths_tr:
        print("dirtyfilePath: " + str(gen_meta(filePath_tr)[0]))
        meta_dirty_tr += gen_meta(filePath_tr)
        print("cleanfilePath: " + str(gen_meta(os.path.join(cleanBasedir,filePath_tr.split('/')[14], filePath_tr.split('/')[15],filePath_tr.split('/')[16]))[0]))
        meta_clean_tr += gen_meta(os.path.join(cleanBasedir,filePath_tr.split('/')[14], filePath_tr.split('/')[15],filePath_tr.split('/')[16]))
        print("---------------------------------------------")


    print("--------------------tt_start--------------------")
    for filePath_tt in filePaths_tt:
        print("dirtyfilePath: " + str(gen_meta(filePath_tt)[0]))
        meta_dirty_tt += gen_meta(filePath_tt)
        print("cleanfilePath: " + str(gen_meta(os.path.join(cleanBasedir,filePath_tt.split('/')[14], filePath_tt.split('/')[15],filePath_tt.split('/')[16]))[0]))
        meta_clean_tt += gen_meta(os.path.join(cleanBasedir,filePath_tt.split('/')[14], filePath_tt.split('/')[15],filePath_tt.split('/')[16]))
        print("---------------------------------------------")

    with open(fileOutPath_tr_noisy, 'w') as f:
        json.dump(meta_dirty_tr, f, indent=4)

    with open(fileOutPath_tr_clean, 'w') as f:
        json.dump(meta_clean_tr, f, indent=4)

    with open(fileOutPath_tt_noisy, 'w') as f:
        json.dump(meta_dirty_tt, f, indent=4)

    with open(fileOutPath_tt_clean, 'w') as f:
        json.dump(meta_clean_tt, f, indent=4)