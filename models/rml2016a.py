import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os
import torchvision

import torch.nn.functional as F
import pickle as pk

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.data import random_split
from collections import Counter

plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Res_18(nn.Module):
    def __init__(self):
        super(Res_18, self).__init__()

        self.conv = nn.Conv2d(1, 3, 1)
        self.resnet = torchvision.models.resnet18(num_classes=11, pretrained=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.resnet(x)

        return x

class Res_50(nn.Module):
    def __init__(self):
        super(Res_50, self).__init__()

        self.conv = nn.Conv2d(1, 3, 1)
        self.resnet = torchvision.models.resnet50(num_classes=11, pretrained=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.resnet(x)

        return x

class Res_34(nn.Module):
    def __init__(self):
        super(Res_34, self).__init__()

        self.conv = nn.Conv2d(1, 3, 1)
        self.resnet = torchvision.models.resnet34(num_classes=11, pretrained=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.resnet(x)

        return x

class VGG_19(nn.Module):
    def __init__(self):
        super(VGG_19, self).__init__()

        self.conv = nn.Conv2d(1, 3, 1)
        self.vgg = torchvision.models.vgg19_bn(num_classes=11, pretrained=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.vgg(x)

        return x

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, groups=1, activation=True):
        super(Conv, self).__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, groups=groups, bias=True)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))

class VGG19(nn.Module):
    def __init__(self, num_classes):
        super(VGG19, self).__init__()
        self.conv = nn.Conv2d(1, 3, 1)
        self.stages = nn.Sequential(*[
            self._make_stage(3, 64, num_blocks=2, max_pooling=True),
            self._make_stage(64, 128, num_blocks=2, max_pooling=True),
            self._make_stage(128, 256, num_blocks=4, max_pooling=True),
            self._make_stage(256, 512, num_blocks=4, max_pooling=True),
            self._make_stage(512, 512, num_blocks=4, max_pooling=True)
        ])
        self.head = nn.Sequential(*[
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1028),
            nn.ReLU(inplace=True),
            nn.Linear(1028, num_classes)
        ])

    @staticmethod
    def _make_stage(in_channels, out_channels, num_blocks, max_pooling):
        layers = [Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)]
        for _ in range(1, num_blocks):
            layers.append(Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        if max_pooling:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.head(self.stages(self.conv(x)))
        # print(x.shape)
        return x

class VTCNN2(nn.Module):
    def __init__(self):
        super(VTCNN2, self).__init__()
        self.model = nn.Sequential(
            nn.ZeroPad2d(padding=(2, 2, 0, 0,)),  # zero pad front/back of each signal by 2
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(1, 3), stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.ZeroPad2d(padding=(2, 2, 0, 0,)),  # zero pad front/back of each signal by 2
            nn.Conv2d(in_channels=256, out_channels=80, kernel_size=(2, 3), stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(in_features=10560, out_features=256, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=11, bias=True),
        )

    def forward(self, x):
        x = self.model(x)

        return x

def get_data():
    xd = pk.load(open('../data/RML2016/RML2016.10a_dict.pkl', 'rb'), encoding='latin1')
    snrs, mods = list(map(lambda j: sorted(list(set(map(lambda x: x[j], xd.keys())))), [1, 0]))
    for snr in snrs:
        x = []
        lbl = []
        for mod in mods:
            x.append(xd[(mod, snr)])
            for i in range(xd[(mod, snr)].shape[0]):
                lbl.append((mod, snr))
        x = np.vstack(x)
        for i in range(x.shape[0]):
            x[i] = (x[i] - x[i].min()) / (x[i].max() - x[i].min())
        y = np.vstack(lbl)

    return x, y

def snr_test(model):
    model.to(device)
    xd = pk.load(open('../data/RML2016/RML2016.10a_dict.pkl', 'rb'), encoding='latin1')
    snrs, mods = list(map(lambda j: sorted(list(set(map(lambda x: x[j], xd.keys())))), [1, 0]))
    accs = []
    for snr in snrs:
        x = []
        lbl = []
        for mod in mods:
            x.append(xd[(mod, snr)])
            for i in range(xd[(mod, snr)].shape[0]):
                lbl.append((mod, snr))
        x = np.vstack(x)
        for i in range(x.shape[0]):
            x[i] = (x[i] - x[i].min()) / (x[i].max() - x[i].min())
        y = np.array([mods.index(i[0]) for i in lbl])
        test_dataset = IQ_dataset(x, y)
        test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
        acc_sum = 0
        loss_sum = 0
        criterion = nn.CrossEntropyLoss(reduction='sum')
        for step, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)

            output, loss, acc = test(model, X, y, criterion)
            loss_sum += loss
            acc_sum += acc

        test_loss = torch.true_divide(loss_sum, x.shape[0])
        test_acc = torch.true_divide(acc_sum, x.shape[0])
        accs.append((test_acc.cpu().item()))
        print(snr, test_loss.item(), test_acc.item())
    plt.plot(snrs, accs)
    plt.xlabel("SNR")
    plt.ylabel("ACC")
    plt.ylim(0, 1.0)
    plt.xticks(snrs)
    plt.yticks(np.arange(0, 1.001, 0.1))
    plt.grid(axis="y", ls='--')
    plt.title("local model")
    plt.show()


class IQ_dataset(Dataset):
    def __init__(self, data, targets):
        super(IQ_dataset, self).__init__()
        self.data = torch.tensor(data)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __getitem__(self, item):
        x = self.data[item]
        y = self.targets[item]
        x = torch.unsqueeze(x, 0)
        return x, y

    def __len__(self):
        return self.data.shape[0]


def accuracy(y_pred, y_true):
    return torch.sum(torch.argmax(y_pred, 1) == y_true).data

def topk_train(model, x, y, k, criterion, optimizer):
    model.train()
    optimizer.zero_grad()

    output = model(x)
    output = torch.softmax(output, 1)

    y = torch.softmax(y, dim=1)
    top_y_value, top_y_indices = y.topk(11, dim=1, )

    y_topk = y[[torch.tensor([[i]*3 for i in range(y.shape[0])]), top_y_indices[:, :k]]]
    output_topk = output[[torch.tensor([[i]*3 for i in range(output.shape[0])]), top_y_indices[:, :k]]]

    loss = criterion(output_topk, y_topk)
    acc = torch.sum(top_y_indices.eq(y) / 11).data

    loss.backward()
    optimizer.step()

    return output, loss, acc


def train(model, X, y, criterion, optimizer):
    model.train()
    optimizer.zero_grad()

    output = model(X)

    top_y_value, top_y_indices = output.topk(11, dim=1,)
    loss = criterion(torch.softmax(output, 1), torch.softmax(y, 1))
    acc = torch.sum(top_y_indices.eq(y)/11).data

    loss.backward()
    optimizer.step()

    return output, loss, acc


@torch.no_grad()
def test(model, X, y, criterion):
    model.eval()

    output = model(X)
    loss = criterion(output, y)
    acc = torch.sum(torch.argmax(output, 1).eq(y)).data

    return output, loss, acc


def IQ_data():
    #https://github.com/radioML/examples/blob/master/modulation_recognition/RML2016.10a_VTCNN2_example.ipynb
    xd = pk.load(open('../data/RML2016/RML2016.10a_dict.pkl', 'rb'), encoding='latin1')
    snrs, mods = list(map(lambda j: sorted(list(set(map(lambda x: x[j], xd.keys())))), [1, 0]))
    x = []
    lbl = []
    for snr in snrs:
        for mod in mods:
            x.append(xd[(mod, snr)])
            for i in range(xd[(mod, snr)].shape[0]):
                lbl.append((mod, snr))
    x = np.vstack(x)
    y = np.array([mods.index(i[0]) for i in lbl])
    for i in range(x.shape[0]):
        x[i] = (x[i]-x[i].min()) / (x[i].max()-x[i].min())
    iq_dataset = IQ_dataset(x, y)
    train_size = int(x.shape[0] * 0.8)
    test_size = int(x.shape[0] * 0.2)
    traindata, testdata = random_split(dataset=iq_dataset, lengths=[train_size, test_size], generator=torch.Generator().manual_seed(0))

    trainloader = DataLoader(traindata, batch_size=256, shuffle=False, num_workers=0)
    testloader = DataLoader(testdata, batch_size=256, shuffle=False, num_workers=0)
    return traindata, testdata, trainloader, testloader

def main(save_path = None):
    # hyperparameters
    epochs = 50
    lr = 0.001

    # init model
    model = VGG19(11)
    model.to(device)

    # dataset & dataloader
    traindata, testdata, trainloader, testloader = IQ_data()
    # _, testdata, _, testloader = IQ_data()
    # indices = testloader.dataset.indices[:1200]
    # testdata = IQ_dataset(torch.squeeze(testdata.dataset.data[indices]), testdata.dataset.targets[indices])
    # testloader = DataLoader(testdata, batch_size=256, shuffle=False, num_workers=0)
    # print(len(testloader.dataset))
    print("count of train dataset: {}\ncount of test dataset: {}".format(len(trainloader.dataset), len(testloader.dataset)))
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False,
                                               threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    min_loss = 1e5
    # trainloader = testloader####
    for epoch in range(1, epochs + 1):
        loss_sum = 0
        acc_num = 0

        train_counter = {}

        for step, (X, y) in enumerate(tqdm(trainloader)):
            X, y = X.to(device), y.to(device)


            pred, loss, acc = train(model, X, y, criterion, optimizer)
            # train_counter = (Counter(torch.argmax(pred, 1).detach().cpu().numpy()) + Counter(train_counter))
            # print(step, "input:", sorted(Counter(y.detach().cpu().numpy()).items()))
            # train_counter = Counter(torch.argmax(pred, 1).detach().cpu().numpy())
            # print(step, "output", sorted(train_counter.items()))
            loss_sum += loss
            acc_num += acc

        val_loss_sum = 0
        val_acc_num = 0
        for val_step, (X_val, y_val) in enumerate(testloader):
            X_val, y_val = X_val.to(device), y_val.to(device)

            output, val_loss, val_acc = test(model, X_val, y_val, criterion)
            val_loss_sum += val_loss
            val_acc_num += val_acc

        if epoch % 1 == 0:
            print("="*50)
            print("Epoch:{}, loss:{}, acc:{}, val_loss:{}, val_acc:{}".format(epoch,
                                                                              torch.true_divide(loss_sum, len(trainloader.dataset)),
                                                                              torch.true_divide(acc_num, len(trainloader.dataset)),
                                                                              torch.true_divide(val_loss_sum, len(testloader.dataset)),
                                                                              torch.true_divide(val_acc_num, len(testloader.dataset))
                                                                              ))
            print("="*50)
        if min_loss > torch.true_divide(loss_sum, len(trainloader.dataset)):
            if not os.path.exists('../saved_models/'):
                os.makedirs('../saved_models/')
            torch.save(model.state_dict(), '../saved_models/RML2016_vgg19.pt')


if __name__ == "__main__":
    main()
