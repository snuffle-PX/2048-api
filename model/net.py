from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn
import numpy as np
import torch.nn.functional as F


# def conv_to_onehont(grid):
#     onehot = np.zeros(grid.shape + (16,),dtype=np.int)
#     for i in range(16):
#         onehot[grid == i] = 1
#     return onehot
#
#
# class Game2048DataSet(Dataset):
#
#     def __init__(self, direction=None):
#         super(Game2048DataSet, self).__init__()
#         self.direction = direction
#         self.data = np.loadtxt(self.direction + "/data_1.csv")
#
#     def __getitem__(self, item):
#         grid = np.reshape(self.data[item, 0:16],(4, 4))
#         target = self.data[item, 16]
#         return conv_to_onehont(grid), target
#
#     def __len__(self):
#         return self.data.__len__()


class nn2048(torch.nn.Module):
    def __init__(self):
        super(nn2048, self).__init__()
        self.model_name = str(type(self))
        self.conv1 = torch.nn.Conv2d(16, 512, kernel_size=(2, 1), padding=0, bias=False)
        self.conv2 = torch.nn.Conv2d(16, 512, kernel_size=(1, 2), padding=0, bias=False)
        self.conv11 = torch.nn.Conv2d(512, 4096, kernel_size=(2, 1), padding=0, bias=True)
        self.conv12 = torch.nn.Conv2d(512, 4096, kernel_size=(1, 2), padding=0, bias=True)
        self.conv21 = torch.nn.Conv2d(512, 4096, kernel_size=(2, 1), padding=0, bias=True)
        self.conv22 = torch.nn.Conv2d(512, 4096, kernel_size=(1, 2), padding=0, bias=True)

        self.fc1 = torch.nn.Linear(34 * 4096, 1000)
        self.fc2 = torch.nn.Linear(1000, 4)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x11 = F.relu(self.conv11(F.relu(x1)))
        x12 = F.relu(self.conv12(F.relu(x1)))
        x21 = F.relu(self.conv21(F.relu(x2)))
        x22 = F.relu(self.conv22(F.relu(x2)))

        x11 = x11.view(-1, self.num_flat_features(x11))
        x12 = x12.view(-1, self.num_flat_features(x12))
        x21 = x21.view(-1, self.num_flat_features(x21))
        x22 = x22.view(-1, self.num_flat_features(x22))

        x = torch.cat([x11, x12, x21, x22], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

    def predict(self, x):
        with torch.no_grad():
            x = self.forward(x)
            preds = torch.argmax(x, dim=1)
        return preds

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    # def load(self):
    #
    # def save(self):


class nn2048_2(torch.nn.Module):
    def __init__(self):
        super(nn2048_2, self).__init__()
        self.model_name = str(type(self))
        self.conv41 = torch.nn.Conv2d(16, 128, kernel_size=(4, 1), padding=0, bias=True)
        self.conv14 = torch.nn.Conv2d(16, 128, kernel_size=(1, 4), padding=0, bias=True)
        self.conv22 = torch.nn.Conv2d(16, 128, kernel_size=(2, 2), padding=0, bias=True)
        self.conv33 = torch.nn.Conv2d(16, 128, kernel_size=(3, 3), padding=0, bias=True)
        self.conv44 = torch.nn.Conv2d(16, 128, kernel_size=(4, 4), padding=0, bias=True)

        # self.norm = torch.nn.BatchNorm1d(128*22)
        self.fc1 = torch.nn.Linear(128 * 22, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 4)

    def forward(self, x):
        x = torch.cat([self.conv14(x).view(x.size()[0], -1),
                       self.conv41(x).view(x.size()[0], -1),
                       self.conv22(x).view(x.size()[0], -1),
                       self.conv33(x).view(x.size()[0], -1),
                       self.conv44(x).view(x.size()[0], -1)], dim=1)
        # x = self.norm(x)

        x = self.fc1(F.relu(x))
        x = self.fc2(F.relu(x))
        x = self.fc3(F.relu(x))

        return x

    def predict(self, x):
        with torch.no_grad():
            x = self.forward(x)
            preds = torch.argmax(x, dim=1)
        return preds

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class nn2048_3(torch.nn.Module):
    def __init__(self):
        super(nn2048_3, self).__init__()
        self.model_name = str(type(self))
        self.conv1 = torch.nn.Conv2d(16, 128, kernel_size=(2, 1), padding=0, bias=True)
        self.conv2 = torch.nn.Conv2d(16, 128, kernel_size=(1, 2), padding=0, bias=True)

        self.conv11 = torch.nn.Conv2d(128, 128, kernel_size=(2, 1), padding=0, bias=True)
        self.conv12 = torch.nn.Conv2d(128, 128, kernel_size=(1, 2), padding=0, bias=True)
        self.conv21 = torch.nn.Conv2d(128, 128, kernel_size=(2, 1), padding=0, bias=True)
        self.conv22 = torch.nn.Conv2d(128, 128, kernel_size=(1, 2), padding=0, bias=True)

        self.fc1 = torch.nn.Linear(128 * 34, 1024)
        self.fc2 = torch.nn.Linear(1024, 128)
        self.fc3 = torch.nn.Linear(128, 4)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x = torch.cat([self.conv11(x1).view(x.size()[0], -1),
                       self.conv12(x1).view(x.size()[0], -1),
                       self.conv21(x2).view(x.size()[0], -1),
                       self.conv22(x2).view(x.size()[0], -1)], dim=1)

        x = self.fc1(F.relu(x))
        x = self.fc2(F.relu(x))
        x = self.fc3(F.relu(x))

        return x

    def predict(self, x):
        with torch.no_grad():
            x = self.forward(x)
            preds = torch.argmax(x, dim=1)
        return preds


class nn2048_4(nn2048_3):
    def __init__(self):
        super(nn2048_4, self).__init__()
        self.conv1 = torch.nn.Conv2d(12, 128, kernel_size=(2, 1), padding=0, bias=True)
        self.conv2 = torch.nn.Conv2d(12, 128, kernel_size=(1, 2), padding=0, bias=True)
