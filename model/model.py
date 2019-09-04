import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output

class SimpleLinear(nn.Module):
    def __init__(self, nIn, nOut):
        super(SimpleLinear, self).__init__()
        self.linear = nn.Linear(nIn, nOut)

    def forward(self, x):
        timesteps, batch_size = x.size(0), x.size(1)
        x = x.view(batch_size*timesteps, -1)
        x = self.linear(x)
        x = x.view(timesteps, batch_size, -1)
        return x

class SimpleLSTM(nn.Module):
    def __init__(self, nIn, nHidden):
        super(SimpleLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)

    def forward(self, input_):
        recurrent, _ = self.rnn(input_)
        T, b, h = recurrent.size()
        return recurrent 
class STN(nn.Module):
    def __init__(self, imgH, imgW):
        super(STN, self).__init__()
        self.imgH = imgH
        self.imgW = imgW
        self.localization = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        h, w = imgH//16, imgW//16
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
        nn.Linear(128 * h * w, 32),
        nn.ReLU(True),
        nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.Tensor([1, 0, 0, 0, 1, 0]).float())

    def forward(self, x):
        xs = self.localization(x)
        b, c, h, w = xs.size(0), xs.size(1), xs.size(2), xs.size(3)
        xs = xs.view(-1, c*h*w)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x    
           
class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, stn_flag=False, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16
        # self.stn_flag = stn_flag
        # if stn_flag:
        #     self.stn = STN(imgH, imgW)
        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        # if self.stn_flag:
        #     input = self.stn(input)
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output

class GravesNet(nn.Module):
    def __init__(self, imgH, nh, nclass, depth):
        super(GravesNet, self).__init__()
        # self.mid_rnn = nn.LSTM(nh, nh, bidirectional=True)
        self.fc_in = SimpleLinear(imgH, nh*2)
        self.hidden_layers = [SimpleLSTM(nh*2, nh)for i in range(depth)]
        # self.hidden_layers = SimpleLSTM(nh*2, nh)
        self.fc_out = SimpleLinear(nh*2, nclass)
        self.module = nn.Sequential(self.fc_in, *self.hidden_layers, self.fc_out)

    def forward(self, input_):
        # conv features
        input_ = input_.squeeze(0)
        input_ = input_.permute(2, 0, 1).contiguous()  # [w, b, c]
        output = self.module(input_)
        return output