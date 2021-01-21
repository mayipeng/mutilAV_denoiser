import os
import cv2
import glob
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from PIL import Image

import AVdataProcess
from model.VASE import *
from videoProcess.get_nets import PNet, RNet, ONet
from videoProcess.detectorPRO import detect_faces_pro
from videoProcess.detector import *
from videoProcess.visualization_utils import *
from videoProcess.my_utils import *
from utils.utils import *

parser = argparse.ArgumentParser(description='EHNET')
parser.add_argument('-L', '--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('-D', '--device', default='0,1,2', type=str,
                    help="Specify the GPU visible in the experiment, e.g. '1,2,3'.")
parser.add_argument("-R", "--resume", action="store_true", default=False,
                    help="Whether to resume training from a recent breakpoint.")
args = parser.parse_args()
basedir_to_save = "/home2/mayipeng/myp_mutilAV/TCDTIMIT/"
basedir = "/home3/zhangzhan/TCDTIMITprocessing/downloadTCDTIMIT/volunteers/01M/Clips/straightcam/"

train_path = "/home2/mayipeng/myp_mutilAV/utils/egs/TCDTIMIT_Babble_5/tr/"
#"/home2/mayipeng/myp_mutilAV/egs/NSDTSEA/tr/"
#"/home2/mayipeng/myp_mutilAV/utils/egs/TCDTIMIT_Babble_5/tr/"
#"/home2/mayipeng/myp_mutilAV/egs/TCDTIMIT/tr/"
test_path = "/home2/mayipeng/myp_mutilAV/utils/egs/TCDTIMIT_Babble_5/tr/"
#"/home2/mayipeng/myp_mutilAV/egs/NSDTSEA/tr/"
#"/home2/mayipeng/myp_mutilAV/utils/egs/TCDTIMIT_Babble_5/tr/"
#"/home2/mayipeng/myp_mutilAV/egs/TCDTIMIT/tr/"
best_enh_sdr = float('-inf')

if args.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

"""
数据加载模块
"""
def data_load(train_path, test_path, basedir_to_save, basedir):
    segment = 4
    stride = 1
    sample_rate = 16000
    length = int(segment * sample_rate)
    stride = int(stride * sample_rate)
    kwargs = {"matching": 'sort', "sample_rate": 16000}
    pad = True
    trainPath = train_path
    train_dataset = AVdataProcess.NoisyCleanSet_onlyAudio(
        trainPath, basedir_to_save, basedir, length=length, stride=stride, pad=pad, **kwargs)
    train_data_loader = DataLoader(
        shuffle=False,
        dataset=train_dataset,
        batch_size=128,
        num_workers=128,
        drop_last=True
    )
    testPath = test_path
    test_dataset = AVdataProcess.NoisyCleanSet_onlyAudio(
        testPath, basedir_to_save, basedir, length=length, stride=stride, pad=pad, **kwargs)
    test_data_loader = DataLoader(
        shuffle=False,
        dataset=test_dataset,
        batch_size=128,
        num_workers=128,
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

    # LOAD VIDEO MODELS
    pnet = PNet()
    rnet = RNet()
    onet = ONet()
    onet.eval()

    v_net = VASE_NET_audio() #VASE_NET_v1()
    v_net = v_net.to(device)
    if device == 'cuda':
        v_net = torch.nn.DataParallel(v_net)
        pnet = torch.nn.DataParallel(pnet)
        rnet = torch.nn.DataParallel(rnet)
        onet = torch.nn.DataParallel(onet)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ASE_NET_v2.pth')
        v_net.load_state_dict(checkpoint['net'])
        best_enh_sdr = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        print("start_epoch: ", start_epoch)

    criterion = 0
    optimizer = optim.Adam(v_net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    return v_net, pnet, rnet, onet, criterion, optimizer, scheduler

# Training
"""
函数名：train()
输入参数：epoch，net（模型参数）, criterion（损失函数参数）, optimizer（优化器参数），trainloader（数据加载器）
返回参数：无
函数功能：训练模型
"""
def train(epoch, va_net, pnet, rnet, onet, optimizer, trainloader, criterion):
    print('\nEpoch: %d' % epoch)
    print('\n-----------------train-----------------')
    train_loss = 0
    va_net.train()
    for batch_idx, (mixture, clean) in enumerate(trainloader):

        """
        VIDEO PROCESS
        """
        print('process video...')
        video_features_b = torch.zeros(128, 120, 256).to(device)  # [16, 120, 256]

        """
        AUDIO PROCESS
        """
        print('process audio...')
        mixture, clean = mixture[1].to(device), clean[1].to(device)

        """
        VASE_NET
        """
        print('training...')
        optimizer.zero_grad()
        outputs = va_net(mixture, video_features_b)
        loss = F.smooth_l1_loss(outputs, clean)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print("batch_idx: ", batch_idx, " | batch_loss: ", loss.item())

    print("\n********** train_loss: ", train_loss/(batch_idx+1))
    return train_loss

"""
函数名：val()
输入参数：epoch，net（模型参数）, criterion（损失函数参数），valloader（数据加载器）
返回参数：无
函数功能：验证与保存模型
"""
def val(epoch, va_net, pnet, rnet, onet, valloader, criterion):
    print('\n-----------------val-----------------')
    global best_enh_sdr
    va_net.eval()
    test_loss = 0.0

    sum_ref_stoi = 0.0
    sum_enh_stoi = 0.0
    sum_ref_sdr = 0.0
    sum_enh_sdr = 0.0
    with torch.no_grad():
        for batch_idx, (mixture, clean) in enumerate(valloader):

            """
            VIDEO PROCESS
            """
            print('process video...')
            video_features_b = torch.zeros(128, 120, 256).to(device)  # [16, 120, 256]

            """
            AUDIO PROCESS
            """
            print('process audio...')
            mixture, clean = mixture[1].to(device), clean[1].to(device)

            """
            VASE_NET
            """
            print('valing...')
            outputs = va_net(mixture, video_features_b)
            loss = F.smooth_l1_loss(outputs, clean)
            test_loss += loss.item()
            print("batch_idx: ", batch_idx, " | batch_loss: ", loss.item())

            predicted = outputs
            targets = clean
            inputs = mixture

            targets = targets.cpu().numpy().reshape(-1,)
            predicted = predicted.cpu().numpy().reshape(-1, )
            inputs = inputs.cpu().numpy().reshape(-1, )

            ref_stoi, enh_stoi, ref_sdr, enh_sdr = object_eval(targets, predicted, inputs)
            sum_ref_stoi += ref_stoi
            sum_enh_stoi += enh_stoi
            sum_ref_sdr += ref_sdr
            sum_enh_sdr += enh_sdr

            # progress_bar(batch_idx, len(valloader), 'val:  Loss: %.3f | ref_stoi: %.3f | enh_stoi: %.3f | ref_sdr: %.3f | enh_sdr: %.3f'
            #     % (test_loss/(batch_idx+1), (sum_ref_stoi/(batch_idx+1)), (sum_enh_stoi/(batch_idx+1)), (sum_ref_sdr/(batch_idx+1)), (sum_enh_sdr/(batch_idx+1))))
        print("val_Loss: ", test_loss/(batch_idx+1), " | ", "ref_stoi: ", sum_ref_stoi/(batch_idx+1), " | ", "enh_stoi: ", sum_enh_stoi/(batch_idx+1), " | ", "ref_sdr: ", sum_ref_sdr/(batch_idx+1), " | ", "enh_sdr: ", sum_enh_sdr/(batch_idx+1))


    # Save checkpoint.
    now_enh_sdr = sum_enh_sdr/len(valloader)
    if now_enh_sdr > best_enh_sdr:
        print('\n-----------------saving-----------------')
        print("best_enh_sdr: ", now_enh_sdr)
        state = {
            'net': va_net.state_dict(),
            'acc': now_enh_sdr,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ASE_v2.pth')
        best_enh_sdr = now_enh_sdr



def main():
    writer = SummaryWriter('logs/train_ASE_v2')
    train_data_loader, test_data_loader = data_load(train_path, test_path, basedir_to_save, basedir)
    v_net, pnet, rnet, onet, criterion, optimizer, scheduler = model_load()
    for epoch in range(450):
        train_loss = train(epoch, v_net, pnet, rnet, onet, optimizer, train_data_loader, criterion)
        val(epoch, v_net, pnet, rnet, onet, test_data_loader, criterion)
        scheduler.step()  # 每隔150 steps学习率乘以0.1
        print('\nlr: %f' % optimizer.param_groups[0]['lr'])
        writer.add_scalar('quadratic', 2**train_loss, global_step=epoch)

if __name__=='__main__':
    print("----------------------------start------------------------------")
    main()


















