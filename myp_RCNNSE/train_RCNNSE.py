import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
# import soundfile as sf
# import librosa
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import CRNN
import dataProcess
from my_utils import progress_bar
from my_utils import object_eval

parser = argparse.ArgumentParser(description='EHNET')
parser.add_argument('-L', '--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('-D', '--device', default='0,1,2', type=str,
                    help="Specify the GPU visible in the experiment, e.g. '1,2,3'.")
parser.add_argument("-R", "--resume", action="store_true", default=True,
                    help="Whether to resume training from a recent breakpoint.")
args = parser.parse_args()
best_enh_sdr = 0

if args.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'


"""
训练数据加载类
Args:
    mixture_dataset (str): mixture dir (wav format files)
    clean_dataset (str): clean dir (wav format files)
    limit (int): the limit of the dataset
    offset (int): the offset of the dataset
"""
# class WavDataset(Dataset):
#     def __init__(self,
#                  mixture_dataset,
#                  clean_dataset,
#                  limit=None,
#                  offset=0,
#                  ):
#
#         mixture_dataset = os.path.abspath(os.path.expanduser(mixture_dataset))
#         clean_dataset = os.path.abspath(os.path.expanduser(clean_dataset))
#         assert os.path.exists(mixture_dataset) and os.path.exists(clean_dataset)
#
#         print("Search datasets...")
#         mixture_wav_files = librosa.util.find_files(mixture_dataset, ext="wav", limit=limit, offset=offset)
#         clean_wav_files = librosa.util.find_files(clean_dataset, ext="wav", limit=limit, offset=offset)
#
#         assert len(mixture_wav_files) == len(clean_wav_files)
#         print(f"\t Original length: {len(mixture_wav_files)}")
#
#         self.length = len(mixture_wav_files)
#         self.mixture_wav_files = mixture_wav_files
#         self.clean_wav_files = clean_wav_files
#
#         print(f"\t Offset: {offset}")
#         print(f"\t Limit: {limit}")
#         print(f"\t Final length: {self.length}")
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, item):
#         mixture_path = self.mixture_wav_files[item]
#         clean_path = self.clean_wav_files[item]
#         name = os.path.splitext(os.path.basename(clean_path))[0]
#
#         mixture, sr = sf.read(mixture_path, dtype="float32")
#         clean, sr = sf.read(clean_path, dtype="float32")
#         assert sr == 16000
#         assert mixture.shape == clean.shape
#
#         n_frames = (len(mixture) - 320) // 160 + 1
#
#         return mixture, clean, n_frames, name

"""
将所有数据补齐到最长
"""
def pad_to_longest(batch):
    mixture_list = []
    clean_list = []
    names = []
    n_frames_list = []
    for mixture, clean, n_frames, name in batch:
        mixture_list.append(torch.tensor(mixture).reshape(-1, 1))
        clean_list.append(torch.tensor(clean).reshape(-1, 1))
        n_frames_list.append(n_frames)
        names.append(name)
    mixture_list = pad_sequence(mixture_list).squeeze(2).permute(1, 0)
    clean_list = pad_sequence(clean_list).squeeze(2).permute(1, 0)

    return mixture_list, clean_list, n_frames_list, names
"""
数据加载模块
"""
def data_load(train_mixture_dir,train_clean_dir,test_mixture_dir,test_clean_dir):
    segment = 4
    stride = 1
    sample_rate = 16000
    length = int(segment * sample_rate)
    stride = int(stride * sample_rate)
    kwargs = {"matching": 'sort', "sample_rate": 16000}
    pad = True
    trainPath = "/home2/mayipeng/denoiser-master/egs/NSDTSEA/tr/"
    train_dataset = dataProcess.NoisyCleanSet(
        trainPath, length=length, stride=stride, pad=pad, **kwargs)
    # train_dataset = WavDataset(train_mixture_dir, train_clean_dir, limit=None, offset=0)
    train_data_loader = DataLoader(
        shuffle=True,
        dataset=train_dataset,
        batch_size=16,
        num_workers=8,
        # collate_fn=pad_to_longest,
        drop_last=True
    )
    testPath = "/home2/mayipeng/denoiser-master/egs/NSDTSEA/tt/"
    test_dataset = dataProcess.NoisyCleanSet(
        testPath, length=length, stride=stride, pad=pad, **kwargs)
    # test_dataset = WavDataset(test_mixture_dir, test_clean_dir, limit=None, offset=0)
    test_data_loader = DataLoader(
        shuffle=True,
        dataset=test_dataset,
        batch_size=16,
        num_workers=32,
        # collate_fn=pad_to_longest,
        drop_last=True
    )
    return train_data_loader, test_data_loader

"""
函数名：model_load()
输入参数：无
返回参数：net（模型参数）, criterion（损失函数参数）, optimizer（优化器参数）, scheduler（自动调参参数）
函数功能：加载模型与参数设置
"""
def model_load():
    # Model
    global best_enh_sdr
    global start_epoch
    print('==> Building model..')

    net = CRNN.CRNN2()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint_1'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint_1/CRNN2.pth')
        net.load_state_dict(checkpoint['net'])
        best_enh_sdr = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        print("start_epoch: ", start_epoch)

    criterion = 0
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
    return net, criterion, optimizer, scheduler

# Training
"""
函数名：train()
输入参数：epoch，net（模型参数）, criterion（损失函数参数）, optimizer（优化器参数），trainloader（数据加载器）
返回参数：无
函数功能：训练模型
"""
def train(epoch, net, optimizer, trainloader, criterion):
    print('\nEpoch: %d' % epoch)
    print('\n--train--')
    net.train()
    train_loss = 0
    # for batch_idx, (mixture, clean, n_frames, name) in enumerate(trainloader):
    for batch_idx, (mixture, clean) in enumerate(trainloader):
        mixture, clean = mixture.to(device), clean.to(device)
        optimizer.zero_grad()
        outputs = net(mixture)
        loss = F.smooth_l1_loss(outputs, clean)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print("batch_idx: ", batch_idx, " | loss: ", loss.item())

    print("********** train_loss: ", train_loss)

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f'
        #     % (train_loss/(batch_idx+1)))

"""
函数名：val()
输入参数：epoch，net（模型参数）, criterion（损失函数参数），valloader（数据加载器）
返回参数：无
函数功能：验证与保存模型
"""
def val(epoch, net, valloader, criterion):
    print('\n--val--')
    global best_enh_sdr
    net.eval()
    test_loss = 0.0

    sum_ref_stoi = 0.0
    sum_enh_stoi = 0.0
    sum_ref_sdr = 0.0
    sum_enh_sdr = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = F.smooth_l1_loss(outputs, targets)
            test_loss += loss.item()
            predicted = outputs

            targets = targets.numpy().reshape(-1,)
            predicted = predicted.numpy().reshape(-1, )
            inputs = inputs.numpy().reshape(-1, )
            ref_stoi, enh_stoi, ref_sdr, enh_sdr = object_eval(targets, predicted, inputs)
            sum_ref_stoi += ref_stoi
            sum_enh_stoi += enh_stoi
            sum_ref_sdr += ref_sdr
            sum_enh_sdr += enh_sdr

            progress_bar(batch_idx, len(valloader), 'val:  Loss: %.3f | ref_stoi: %.3f | enh_stoi: %.3f | ref_sdr: %.3f | enh_sdr: %.3f'
                % (test_loss/(batch_idx+1), (sum_ref_stoi/(batch_idx+1)), (sum_enh_stoi/(batch_idx+1)), (sum_ref_sdr/(batch_idx+1)), (sum_enh_sdr/(batch_idx+1))))

    # Save checkpoint.
    now_enh_sdr = sum_enh_sdr/len(valloader)
    if now_enh_sdr > best_enh_sdr:
        print('---------------------------------------------------------------------------')
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': now_enh_sdr,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint_1'):
            os.mkdir('checkpoint_1')
        torch.save(state, './checkpoint_1/CRNN2.pth')
        best_enh_sdr = now_enh_sdr

def main():
    train_mixture_dir = "/home2/mayipeng/phasen-master/data/NSDTSEA_16k/noisy/train/"
    train_clean_dir = "/home2/mayipeng/phasen-master/data/NSDTSEA_16k/clean/train/"
    test_mixture_dir = "/home2/mayipeng/phasen-master/data/NSDTSEA_16k/noisy/test/"
    test_clean_dir = "/home2/mayipeng/phasen-master/data/NSDTSEA_16k/clean/test/"

    train_data_loader, test_data_loader = data_load(train_mixture_dir, train_clean_dir, test_mixture_dir, test_clean_dir)
    net, criterion, optimizer, scheduler = model_load()

    for epoch in range(450):
        train(epoch, net, optimizer, train_data_loader, criterion)
        val(epoch, net, test_data_loader, criterion)
        scheduler.step()  # 每隔150 steps学习率乘以0.1
        print('lr: %f' % optimizer.param_groups[0]['lr'])

if __name__=='__main__':
    main()