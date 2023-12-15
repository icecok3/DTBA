import torch
import torch.nn as nn
import torchvision
import numpy as np
import rml2016a
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False 

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary
from tqdm import tqdm

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


def jacobian(model, x, class_num):
    '''
    compute the Jacobian matrix
    :param model: Substitute model
    :param x: Current data set
    :param class_num: numbers of the kinds of tragets
    :return: jacobian matrix
    '''
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
    '''
    Jacobian-based Dataset Augmentation
    :param model:
    :param X:
    :param lmbda:
    :return:
    '''
    x_length = X.shape[0]
    X = torch.cat((X, X), 0)
    for i in range(x_length):
        y_cur, grad = jacobian(model, X[i], 11)
        grad = grad[y_cur.item()]
        grad = np.sign(grad)
        X[x_length + i] = torch.unsqueeze(X[i] + lmbda*grad, 0).cpu()

    return X, torch.zeros(X.shape[0], )


def Black_Box_Attack(attack_model, data_augment_epochs=5, random_size=3, jacob_rate = 0.2, save_path=None):
    assert save_path != None
    # hyperparameters
    train_epoches = 20
    lr = 0.001
    batch_size = 150

    # load vivtim model
    victim_model = rml2016a.VTCNN2()
    victim_model.to(device)
    victim_model.load_state_dict(torch.load('../saved_models/RML2016.pt'))

    # choose substitute module
    attacker_model = attack_model
    attacker_model.to(device)

    # choose data set
    _, _, _, valid_loader = rml2016a.IQ_data()
    import pickle as pk
    import itertools
    xd = pk.load(open('../data/RML2016/RML2016.10a_dict.pkl', 'rb'), encoding='latin1')
    snrs, mods = list(map(lambda j: sorted(list(set(map(lambda x: x[j], xd.keys())))), [1, 0]))
    x = []
    y = []
    lbl = []
    snr_ = snrs
    query_num = 0

    for i in itertools.product(mods, snr_):
        x_ = xd[i]
        np.random.seed(0)
        x_ = x_[np.random.choice(np.arange(0, x_.shape[0]), random_size, replace=False)]
        x.append(x_)
        # x.append(np.expand_dims(x_, 0))
        y.append([mods.index(i[0])]*random_size)

    x = np.vstack(x)
    y = np.hstack(y)
    for i in range(x.shape[0]):
        x[i] = (x[i] - x[i].min()) / (x[i].max() - x[i].min())
    testdata = rml2016a.IQ_dataset(x, y)
    testloader = DataLoader(testdata, batch_size=256, shuffle=False, num_workers=0)

    # criterion = nn.CrossEntropyLoss(reduction='sum')
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(attacker_model.parameters(), lr)

    min_loss = 1e5
    for data_augment_epoch in range(data_augment_epochs):
        print("data augment:", data_augment_epoch)

        for train_epoch in range(train_epoches):
            att_loss = 0
            for step, (x, y) in enumerate((testloader)):
                x, y = x.to(device), y.to(device)

                # substitute dataset labeling
                victim_model.eval()
                vic_output = victim_model(x)

                # substitute model training
                att_output, al, _ = rml2016a.topk_train(attacker_model, x, vic_output, 3,
                                                   criterion, optimizer)
                att_loss += al

            att_loss = torch.true_divide(att_loss, len(testloader.dataset)).item()
            if att_loss < min_loss and train_epoch != 0:
                min_loss = att_loss
                torch.save(attacker_model.state_dict(), save_path)
            print("\ttrain epoch:", train_epoch, ", att_loss:", att_loss)
        print("\n\t######### data set shape:", len(testloader.dataset), "#########")

        query_num += len(testloader.dataset)

        # dataset augmentation
        if data_augment_epoch == data_augment_epochs - 1:
            break
        print("\tdataset augmenting...")
        x_augmented, y_augmented = jacobian_augmentation(attacker_model, testdata.data, jacob_rate)

        testdata.data = torch.squeeze(x_augmented).cpu()
        testdata.targets = y_augmented.cpu()
        print("\t######### augmented data set shape:", testdata.data.shape[0], " #########\n")

        testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True, num_workers=0)



    del victim_model, attack_model, x, y
    torch.cuda.empty_cache()

    return query_num

