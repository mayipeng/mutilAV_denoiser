#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

path=egs/TCDTIMIT_Babble_5/tr
mkdir -p $path

python3 -m denoiser.audio /home2/mayipeng/myp_mutilAV/data/audio/u/drspeech/data/TCDTIMIT/Noisy_TCDTIMIT/Babble/5/volunteers/01M/straightcam > $path/noisy.json
python3 -m denoiser.audio /home2/mayipeng/myp_mutilAV/data/audio/Clean/volunteers/01M/straightcam > $path/clean.json

