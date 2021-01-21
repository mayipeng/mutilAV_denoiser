from torch import nn
import torch
import math
from torch.nn import functional as F


class CRNN2(nn.Module):
    def __init__(self):
        super().__init__()

        self.E1 = nn.Conv1d(1, 2, 8, 4)
        self.E2 = nn.Conv1d(2, 4, 1)
        self.E3 = nn.Conv1d(4, 8, 8, 4)
        self.E4 = nn.Conv1d(8, 16, 1)
        self.E5 = nn.Conv1d(16, 32, 8, 4)
        self.E6 = nn.Conv1d(32, 64, 1)
        self.E7 = nn.Conv1d(64, 128, 8, 4)
        self.E8 = nn.Conv1d(128, 256, 1)
        self.E9 = nn.Conv1d(256, 512, 8, 4)
        self.E10 = nn.Conv1d(512, 1024, 1)
        self.R = nn.ReLU()

        self.LSTM = nn.LSTM(input_size=62, hidden_size=62, num_layers=2, batch_first=True)

        self.D1 = nn.Conv1d(1024, 512, 1)
        self.D2 = nn.ConvTranspose1d(512, 256, 8, 4)
        self.D3 = nn.Conv1d(256, 128, 1)
        self.D4 = nn.ConvTranspose1d(128, 64, 8, 4)
        self.D5 = nn.Conv1d(64, 32, 1)
        self.D6 = nn.ConvTranspose1d(32, 16, 8, 4)
        self.D7 = nn.Conv1d(16, 8, 1)
        self.D8 = nn.ConvTranspose1d(8, 4, 8, 4)
        self.D9 = nn.Conv1d(4, 2, 1)
        self.D10 = nn.ConvTranspose1d(2, 1, 8, 4)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length)
        for idx in range(5):
            length = math.ceil((length - 8) / 4) + 1
            length = max(length, 1)
        for idx in range(5):
            length = (length - 1) * 4 + 8
        length = int(math.ceil(length))
        return int(length)

    def forward(self, x):
        # print(x.size())
        # x = torch.unsqueeze(x, 2)
        # print(x.size())
        # x = x.permute(0, 2, 1)
        # print("a", x.size())
        mono = x.mean(dim=1, keepdim=True)
        std = mono.std(dim=-1, keepdim=True)
        x = x / (1e-3 + std)
        length = x.shape[-1]
        x = F.pad(x, (0, self.valid_length(length) - length))
        # c = F.pad(c, (0, self.valid_length(length) - length))
        # print(x.size())

        x0 = self.E1(x)
        x1 = self.R(x0)
        x2 = self.E2(x1)


        x3 = self.E3(x2)
        x3 = self.R(x3)
        x4 = self.E4(x3)


        x5 = self.E5(x4)
        x5 = self.R(x5)
        x6 = self.E6(x5)


        x7 = self.E7(x6)
        x7 = self.R(x7)
        x8 = self.E8(x7)


        x9 = self.E9(x8)
        x9 = self.R(x9)
        x10 = self.E10(x9)

        x11, _ = self.LSTM(x10)

        x = self.D1(x11) + x9
        x = self.R(x)
        x = self.D2(x) + x8


        x = self.D3(x) + x7
        x = self.R(x)
        x = self.D4(x) + x6


        x = self.D5(x) + x5
        x = self.R(x)
        x = self.D6(x) + x4


        x = self.D7(x) + x3
        x = self.R(x)
        x = self.D8(x) + x2


        x = self.D9(x) + x1
        x = self.R(x)
        x = self.D10(x)

        # print("x1: " + str(x1.size()))
        # print("x2: " + str(x2.size()))
        # print("x3: " + str(x3.size()))
        # print("x4: " + str(x4.size()))
        # print("x5: " + str(x5.size()))
        # print("x6: " + str(x6.size()))
        # print("x7: " + str(x7.size()))
        # print("x8: " + str(x8.size()))
        # print("x9: " + str(x9.size()))
        # print("x10: " + str(x10.size()))
        # print("xd1: " + str(x.size()))
        # print("xd2: " + str(x.size()))
        # print("xd3: " + str(x.size()))
        # print("xd4: " + str(x.size()))
        # print("xd5: " + str(x.size()))
        # print("xd6: " + str(x.size()))
        # print("xd7: " + str(x.size()))
        # print("xd8: " + str(x.size()))
        # print("xd9: " + str(x.size()))
        # print("xd10: " + str(x.size()))

        ############################################
        # x = self.encoder(x)
        # print(x.size())
        # # print(x.size())
        # # x = x.view(x.size(0), 256 * 2 * 2)
        # x = self.decoder(x)
        # # print(c.size())
        ############################################
        x = x[..., :length]
        return std * x



class VASE_NET_v1(nn.Module):
    def __init__(self):
        super().__init__()

        self.E1 = nn.Conv1d(1, 2, 8, 4)
        self.E2 = nn.Conv1d(2, 4, 1)
        self.E3 = nn.Conv1d(4, 8, 8, 4)
        self.E4 = nn.Conv1d(8, 16, 1)
        self.E5 = nn.Conv1d(16, 32, 8, 4)
        self.E6 = nn.Conv1d(32, 64, 1)
        self.E7 = nn.Conv1d(64, 128, 8, 4)
        self.E8 = nn.Conv1d(128, 256, 1)
        self.E9 = nn.Conv1d(256, 512, 8, 4)
        self.E10 = nn.Conv1d(512, 1024, 1)
        self.R = nn.ReLU()

        self.VE1 = nn.Linear(30720, 1024)
        self.VE2 = nn.Linear(1024, 31744)

        self.S1 = nn.Conv1d(1024, 1024, 2, 2)

        self.LSTM = nn.LSTM(input_size=62, hidden_size=62, num_layers=2, batch_first=True)

        self.D1 = nn.Conv1d(1024, 512, 1)
        self.D2 = nn.ConvTranspose1d(512, 256, 8, 4)
        self.D3 = nn.Conv1d(256, 128, 1)
        self.D4 = nn.ConvTranspose1d(128, 64, 8, 4)
        self.D5 = nn.Conv1d(64, 32, 1)
        self.D6 = nn.ConvTranspose1d(32, 16, 8, 4)
        self.D7 = nn.Conv1d(16, 8, 1)
        self.D8 = nn.ConvTranspose1d(8, 4, 8, 4)
        self.D9 = nn.Conv1d(4, 2, 1)
        self.D10 = nn.ConvTranspose1d(2, 1, 8, 4)

    def valid_length(self, length):
        length = math.ceil(length)
        for idx in range(5):
            length = math.ceil((length - 8) / 4) + 1
            length = max(length, 1)
        for idx in range(5):
            length = (length - 1) * 4 + 8
        length = int(math.ceil(length))
        return int(length)

    def forward(self, x, v):
        v = v.reshape(-1, 30720)
        # print(x.size())
        # print(v.size())
        mono = x.mean(dim=1, keepdim=True)
        std = mono.std(dim=-1, keepdim=True)
        x = x / (1e-3 + std)
        length = x.shape[-1]
        x = F.pad(x, (0, self.valid_length(length) - length))

        v1 = self.VE1(v)
        v1 = self.R(v1)
        v2 = self.VE2(v1)
        v2 = self.R(v2)
        v2 = v2.reshape(-1, 1024, 31)

        x0 = self.E1(x)
        x1 = self.R(x0)
        x2 = self.E2(x1)


        x3 = self.E3(x2)
        x3 = self.R(x3)
        x4 = self.E4(x3)


        x5 = self.E5(x4)
        x5 = self.R(x5)
        x6 = self.E6(x5)


        x7 = self.E7(x6)
        x7 = self.R(x7)
        x8 = self.E8(x7)


        x9 = self.E9(x8)
        x9 = self.R(x9)
        x10 = self.E10(x9)

        x101 = self.S1(x10)
        x102 = torch.cat((x101, v2), 2)

        x11, _ = self.LSTM(x102)

        x12 = self.D1(x11) + x9
        x12 = self.R(x12)
        x13 = self.D2(x12) + x8


        x14 = self.D3(x13) + x7
        x14 = self.R(x14)
        x15 = self.D4(x14) + x6


        x16 = self.D5(x15) + x5
        x16 = self.R(x16)
        x17 = self.D6(x16) + x4


        x18 = self.D7(x17) + x3
        x18 = self.R(x18)
        x19 = self.D8(x18) + x2


        x20 = self.D9(x19) + x1
        x20 = self.R(x20)
        x21 = self.D10(x20)

        # print("x1: " + str(x1.size()))
        # print("x2: " + str(x2.size()))
        # print("x3: " + str(x3.size()))
        # print("x4: " + str(x4.size()))
        # print("x5: " + str(x5.size()))
        # print("x6: " + str(x6.size()))
        # print("x7: " + str(x7.size()))
        # print("x8: " + str(x8.size()))
        # print("x9: " + str(x9.size()))
        # print("x10: " + str(x10.size()))
        # print("xd1: " + str(x11.size()))
        # print("xd2: " + str(x12.size()))
        # print("xd3: " + str(x13.size()))
        # print("xd4: " + str(x14.size()))
        # print("xd5: " + str(x15.size()))
        # print("xd6: " + str(x16.size()))
        # print("xd7: " + str(x17.size()))
        # print("xd8: " + str(x18.size()))
        # print("xd9: " + str(x19.size()))
        # print("xd10: " + str(x20.size()))

        ############################################
        # x = self.encoder(x)
        # print(x.size())
        # # print(x.size())
        # # x = x.view(x.size(0), 256 * 2 * 2)
        # x = self.decoder(x)
        # # print(c.size())
        ############################################
        x21 = x21[..., :length]
        return std * x21

class VASE_NET_audio(nn.Module):
    def __init__(self):
        super().__init__()

        self.E1 = nn.Conv1d(1, 2, 8, 4)
        self.E2 = nn.Conv1d(2, 4, 1)
        self.E3 = nn.Conv1d(4, 8, 8, 4)
        self.E4 = nn.Conv1d(8, 16, 1)
        self.E5 = nn.Conv1d(16, 32, 8, 4)
        self.E6 = nn.Conv1d(32, 64, 1)
        self.E7 = nn.Conv1d(64, 128, 8, 4)
        self.E8 = nn.Conv1d(128, 256, 1)
        self.E9 = nn.Conv1d(256, 512, 8, 4)
        self.E10 = nn.Conv1d(512, 1024, 1)
        self.R = nn.ReLU()

        self.VE1 = nn.Linear(30720, 1024)
        self.VE2 = nn.Linear(1024, 31744)

        self.S1 = nn.Conv1d(1024, 1024, 2, 2)

        self.LSTM = nn.LSTM(input_size=62, hidden_size=62, num_layers=2, batch_first=True)

        self.D1 = nn.Conv1d(1024, 512, 1)
        self.D2 = nn.ConvTranspose1d(512, 256, 8, 4)
        self.D3 = nn.Conv1d(256, 128, 1)
        self.D4 = nn.ConvTranspose1d(128, 64, 8, 4)
        self.D5 = nn.Conv1d(64, 32, 1)
        self.D6 = nn.ConvTranspose1d(32, 16, 8, 4)
        self.D7 = nn.Conv1d(16, 8, 1)
        self.D8 = nn.ConvTranspose1d(8, 4, 8, 4)
        self.D9 = nn.Conv1d(4, 2, 1)
        self.D10 = nn.ConvTranspose1d(2, 1, 8, 4)

    def valid_length(self, length):
        length = math.ceil(length)
        for idx in range(5):
            length = math.ceil((length - 8) / 4) + 1
            length = max(length, 1)
        for idx in range(5):
            length = (length - 1) * 4 + 8
        length = int(math.ceil(length))
        return int(length)

    def forward(self, x, v):
        v = v.reshape(-1, 30720)
        # print(x.size())
        # print(v.size())
        mono = x.mean(dim=1, keepdim=True)
        std = mono.std(dim=-1, keepdim=True)
        x = x / (1e-3 + std)
        length = x.shape[-1]
        x = F.pad(x, (0, self.valid_length(length) - length))

        v1 = self.VE1(v)
        v1 = self.R(v1)
        v2 = self.VE2(v1)
        v2 = self.R(v2)
        v2 = v2.reshape(-1, 1024, 31)

        v2 = torch.full(v2.size(), 0).to('cuda')
        # v2 = torch.zeros(128, 1024, 31).to('cuda')

        x0 = self.E1(x)
        x1 = self.R(x0)
        x2 = self.E2(x1)


        x3 = self.E3(x2)
        x3 = self.R(x3)
        x4 = self.E4(x3)


        x5 = self.E5(x4)
        x5 = self.R(x5)
        x6 = self.E6(x5)


        x7 = self.E7(x6)
        x7 = self.R(x7)
        x8 = self.E8(x7)


        x9 = self.E9(x8)
        x9 = self.R(x9)
        x10 = self.E10(x9)

        x101 = self.S1(x10)
        x102 = torch.cat((x101, v2), 2)

        x11, _ = self.LSTM(x102)

        x12 = self.D1(x11) + x9
        x12 = self.R(x12)
        x13 = self.D2(x12) + x8


        x14 = self.D3(x13) + x7
        x14 = self.R(x14)
        x15 = self.D4(x14) + x6


        x16 = self.D5(x15) + x5
        x16 = self.R(x16)
        x17 = self.D6(x16) + x4


        x18 = self.D7(x17) + x3
        x18 = self.R(x18)
        x19 = self.D8(x18) + x2


        x20 = self.D9(x19) + x1
        x20 = self.R(x20)
        x21 = self.D10(x20)

        # print("x1: " + str(x1.size()))
        # print("x2: " + str(x2.size()))
        # print("x3: " + str(x3.size()))
        # print("x4: " + str(x4.size()))
        # print("x5: " + str(x5.size()))
        # print("x6: " + str(x6.size()))
        # print("x7: " + str(x7.size()))
        # print("x8: " + str(x8.size()))
        # print("x9: " + str(x9.size()))
        # print("x10: " + str(x10.size()))
        # print("xd1: " + str(x11.size()))
        # print("xd2: " + str(x12.size()))
        # print("xd3: " + str(x13.size()))
        # print("xd4: " + str(x14.size()))
        # print("xd5: " + str(x15.size()))
        # print("xd6: " + str(x16.size()))
        # print("xd7: " + str(x17.size()))
        # print("xd8: " + str(x18.size()))
        # print("xd9: " + str(x19.size()))
        # print("xd10: " + str(x20.size()))
        # print("xd102: " + str(x102.size()))
        # print("v2: " + str(v2.size()))

        ############################################
        # x = self.encoder(x)
        # print(x.size())
        # # print(x.size())
        # # x = x.view(x.size(0), 256 * 2 * 2)
        # x = self.decoder(x)
        # # print(c.size())
        ############################################
        x21 = x21[..., :length]
        return std * x21

if __name__ == "__main__":
    input1 = torch.randn(128, 1, 64000)
    input2 = torch.randn(128, 120, 256)
    model = VASE_NET_audio()
    out = model(input1, input2)
    print(out.size())















