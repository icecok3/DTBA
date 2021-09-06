import torch
import torch.nn as nn
import torchvision
import numpy as np
import MNIST
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary
from tqdm import tqdm
import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_size = 150

class attacker(nn.Module):
    def __init__(self):
        super(attacker, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(1, 3, 1),
                                   torchvision.models.resnet18(pretrained=False),
                                   nn.Linear(1000, 10)
                                   )

    def forward(self, x):
        x = self.model(x)

        return x

def fgsm(model, x, y, criterion, is_target=None, eps=0.1):
    x_adv = x
    # x.to(device)
    x_adv.requires_grad = True

    output = model(x)
    loss = criterion(output, y)
    model.zero_grad()
    loss.backward()
    x_grad_sign = x_adv.grad.data.sign()
    x_adv = x_adv + eps * x_grad_sign
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv, model(x_adv)

def jsma():
    pass

def jacobian(model, x, class_num):
    """
    compute the Jacobian matrix
    :param model: Substitute model
    :param x: Current data set
    :param class_num: numbers of the kinds of tragets
    :return: jacobian matrix
    """
    model.eval()
    gradient_matrix = []
    x = x.type(torch.FloatTensor).to(device)
    x.requires_grad = True
    for class_idx in range(class_num):
        output = model(torch.unsqueeze(torch.unsqueeze(x, 0), 0))
        y = torch.argmax(output, 1)
        label_grad = output[0, class_idx]
        label_grad.backward()
        gradient_matrix.append(x.grad.data.cpu().numpy())
        x.grad.data.zero_()

    return y, gradient_matrix

def jacobian_augmentation(model, X, lmbda=0.1):
    """
    Jacobian-based Dataset Augmentation
    :param model:
    :param X:
    :param lmbda:
    :return:
    """
    x_length = X.shape[0]
    X = torch.cat((X, X), 0)
    for i in range(x_length):
        y_cur, grad = jacobian(model, X[i], 10)
        grad = grad[y_cur.item()]
        grad = np.sign(grad)
        # X = torch.vstack((X, torch.unsqueeze(x_cur + lmbda*grad, 0).cpu()))
        X[x_length + i] = torch.unsqueeze(X[i] + lmbda * grad, 0).cpu()

    return X, torch.zeros(X.shape[0], )


def main():
    # hyperparameters
    data_augment_epochs = 6
    train_epoches = 10
    lr = 0.01
    batch_size = 150

    # load victim model
    victim_model = MNIST.MNIST()
    victim_model.to(device)
    victim_model.load_state_dict(torch.load('../saved_models/MNIST.pt'))

    # choose substitute module
    attacker_model = attacker()
    attacker_model.to(device)

    # choose data set
    mnist_transform = transforms.Compose([transforms.ToTensor()])
    testdata = torchvision.datasets.MNIST(root="../data", train=False, transform=mnist_transform)
    testdata.data = testdata.data[:data_size]
    testdata.targets = testdata.targets[:data_size]
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(attacker_model.parameters(), lr)

    min_loss = 1e5
    for data_augment_epoch in range(data_augment_epochs):
        print("data augment:", data_augment_epoch)
        for train_epoch in range(train_epoches):
            att_loss = 0
            for step, (x, y) in enumerate((testloader)):
                x, y = x.to(device), y.to(device)
                # step 3 substitute dataset labeling
                victim_model.eval()
                vic_output = victim_model(x)
                # step4 substitute model training
                att_output, al, _ = MNIST.train(attacker_model, x, torch.argmax(vic_output, 1), criterion, optimizer)
                att_loss += al

            att_loss = torch.true_divide(att_loss, len(testloader.dataset)).item()
            if att_loss < min_loss and train_epoch != 0:
                min_loss = att_loss
                torch.save(attacker_model.state_dict(), '../saved_models/MNIST_SUB.pt')
            print("\ttrain epoch:", train_epoch, ", att_loss:", att_loss)
        print("\n\t######### data set shape:", len(testloader.dataset), "#########")

        # step 5 dataset augmentation
        if data_augment_epoch == data_augment_epochs - 1:
            break
        print("\tdataset augmenting...")
        x_augmented, y_augmented = jacobian_augmentation(attacker_model, testdata.data)
        testdata.data = torch.squeeze(x_augmented).cpu()
        testdata.targets = y_augmented.cpu()
        print("\t######### augmented data set shape:", testdata.data.shape[0], " #########\n")
        testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=0)


if __name__ == "__main__":
    main()
