import json
from pathlib import Path
import math
import os
import sys
import re
import logging

import torchaudio
from torch.nn import functional as F

class Audioset:
    def __init__(self, files=None, length=None, stride=None,
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
        for file, file_length in self.files:
            if length is None:
                examples = 1
            elif file_length < length:
                examples = 1 if pad else 0
            elif pad:
                examples = int(math.ceil((file_length - self.length) / self.stride) + 1)
            else:
                examples = (file_length - self.length) // self.stride + 1
            self.num_examples.append(examples)  # 64000是一段，如果长度大于64000的话，每次向后走16000，examples是段数
            print(file_length, length, self.stride, examples)

    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        for (file, _), examples in zip(self.files, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            num_frames = 0
            offset = 0
            if self.length is not None:
                offset = self.stride * index
                num_frames = self.length
            out, sr = torchaudio.load(str(file), offset=offset, num_frames=num_frames)
            if self.sample_rate is not None:
                if sr != self.sample_rate:
                    raise RuntimeError(f"Expected {file} to have sample rate of "
                                       f"{self.sample_rate}, but got {sr}")
            if num_frames:
                out = F.pad(out, (0, num_frames - out.shape[-1]))
            if self.with_path:
                return out, file
            else:
                return out

logger = logging.getLogger(__name__)


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


class NoisyCleanSet:
    def __init__(self, json_dir, matching="sort", length=None, stride=None,
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
        self.clean_set = Audioset(clean, **kw)
        self.noisy_set = Audioset(noisy, **kw)
        print(len(self.clean_set))
        print(len(self.noisy_set))
        assert len(self.clean_set) == len(self.noisy_set)

    def __getitem__(self, index):
        return self.noisy_set[index], self.clean_set[index]

    def __len__(self):
        return len(self.noisy_set)




if __name__=='__main__':
    segment = 4
    stride = 1
    sample_rate = 16000
    length = int(segment * sample_rate)
    stride = int(stride * sample_rate)
    kwargs = {"matching": 'sort', "sample_rate": 16000}
    pad = True
    trainPath = "/home2/mayipeng/denoiser-master/egs/NSDTSEA/tr/"
    tr_dataset = NoisyCleanSet(
        trainPath, length=length, stride=stride, pad=pad, **kwargs)
    for a, b in tr_dataset:
        print(a.size(), b.size())










