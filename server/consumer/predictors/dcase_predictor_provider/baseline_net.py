"""This module contains the neural network architecture used
by the module ``dcase_predictor_provider``

"""
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):


    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 5, stride = 2, padding = 2)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, stride = 1, padding = 1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.do1 = nn.Dropout2d(0.3)

        self.conv3 = nn.Conv2d(64, 128, 3, stride = 1, padding = 1)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride = 1, padding = 1)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.do2 = nn.Dropout2d(0.3)

        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.do3 = nn.Dropout2d(0.3)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv6_bn = nn.BatchNorm2d(256)
        self.do4 = nn.Dropout2d(0.3)
        self.conv7 = nn.Conv2d(256, 384, 3, stride=1, padding=1)
        self.conv7_bn = nn.BatchNorm2d(384)
        self.do5 = nn.Dropout2d(0.3)
        self.conv8 = nn.Conv2d(384, 384, 3, stride=1, padding=1)
        self.conv8_bn = nn.BatchNorm2d(384)
        self.do6 = nn.Dropout2d(0.3)

        self.conv9 = nn.Conv2d(384, 512, 3, stride=1, padding=1)
        self.conv9_bn = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv10_bn = nn.BatchNorm2d(512)
        self.do7 = nn.Dropout2d(0.3)

        self.conv11 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv11_bn = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv12_bn = nn.BatchNorm2d(512)
        self.do8 = nn.Dropout2d(0.3)

        self.conv13 = nn.Conv2d(512, 512, 3, stride=1, padding=0)
        self.conv13_bn = nn.BatchNorm2d(512)
        self.do9 = nn.Dropout2d(0.5)
        self.conv14 = nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.conv14_bn = nn.BatchNorm2d(512)
        self.do10 = nn.Dropout2d(0.5)
        self.conv15 = nn.Conv2d(512, 41, 1, stride=1, padding=0)
        self.conv15_bn = nn.BatchNorm2d(41)

        self.initialize_weights(self.conv1)
        self.initialize_weights(self.conv1_bn)
        self.initialize_weights(self.conv2)
        self.initialize_weights(self.conv2_bn)
        self.initialize_weights(self.conv3)
        self.initialize_weights(self.conv3_bn)
        self.initialize_weights(self.conv4)
        self.initialize_weights(self.conv4_bn)
        self.initialize_weights(self.conv5)
        self.initialize_weights(self.conv5_bn)
        self.initialize_weights(self.conv6)
        self.initialize_weights(self.conv6_bn)
        self.initialize_weights(self.conv7)
        self.initialize_weights(self.conv7_bn)
        self.initialize_weights(self.conv8)
        self.initialize_weights(self.conv8_bn)
        self.initialize_weights(self.conv9)
        self.initialize_weights(self.conv9_bn)
        self.initialize_weights(self.conv10)
        self.initialize_weights(self.conv10_bn)
        self.initialize_weights(self.conv11)
        self.initialize_weights(self.conv11_bn)
        self.initialize_weights(self.conv12)
        self.initialize_weights(self.conv12_bn)
        self.initialize_weights(self.conv13)
        self.initialize_weights(self.conv13_bn)
        self.initialize_weights(self.conv14)
        self.initialize_weights(self.conv14_bn)
        self.initialize_weights(self.conv15)
        self.initialize_weights(self.conv15_bn)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        #print(x.size())
        x = F.relu(self.conv2_bn(self.conv2(x)))
        #print(x.size())
        x = F.max_pool2d(x, (2, 2))
        #print(x.size())
        x = self.do1(x)

        x = F.relu(self.conv3_bn(self.conv3(x)))
        #print(x.size())
        x = F.relu(self.conv4_bn(self.conv4(x)))
        #print(x.size())
        x = F.max_pool2d(x, (2, 2))
        #print(x.size())
        x = self.do2(x)

        x = F.relu(self.conv5_bn(self.conv5(x)))
        #print(x.size())
        x = self.do3(x)
        x = F.relu(self.conv6_bn(self.conv6(x)))
        #print(x.size())
        x = self.do4(x)
        x = F.relu(self.conv7_bn(self.conv7(x)))
        #print(x.size())
        x = self.do5(x)
        x = F.relu(self.conv8_bn(self.conv8(x)))
        #print(x.size())
        x = F.max_pool2d(x, (2, 2))
        #print(x.size())
        x = self.do6(x)

        x = F.relu(self.conv9_bn(self.conv9(x)))
        #print(x.size())
        x = F.relu(self.conv10_bn(self.conv10(x)))
        #print(x.size())
        x = F.max_pool2d(x, (1, 2))
        #print(x.size())
        x = self.do7(x)

        x = F.relu(self.conv11_bn(self.conv11(x)))
        #print(x.size())
        x = F.relu(self.conv12_bn(self.conv12(x)))
        #print(x.size())
        x = F.max_pool2d(x, (1, 2))
        #print(x.size())
        x = self.do8(x)

        x = F.relu(self.conv13_bn(self.conv13(x)))
        #print(x.size())
        x = self.do9(x)
        x = F.relu(self.conv14_bn(self.conv14(x)))
        #print(x.size())
        x = self.do10(x)

        x = self.conv15_bn(self.conv15(x))
        #print(x.size())
        size_to_pool = x.size()
        x = F.avg_pool2d(x, (size_to_pool[2],size_to_pool[3]))
        x = x.view(-1, 41)
        #print(x.size())
        return x

    def initialize_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity="relu")
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.uniform_()
            module.bias.data.zero_()
        elif isinstance(module, nn.Linear):
            module.bias.data.zero_()