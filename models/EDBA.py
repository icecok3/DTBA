import torch
import torch.nn as nn
import torch.nn.functional as F
import rml2016a
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import Parameter
import math
plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        output = (cos_theta,phi_theta)
        return output # size=(B,Classnum,2)

def criterion(net, x, x_adv):
    # x, x_adv : torch.Tensor with shape [1,3,112,96]
    best_thresh = 0.3

    imglist = [x, x_adv]
    img = torch.vstack(imglist)
    img = Variable(img.float(), volatile=True).cuda()
    #####
    output = net(img)
    angl = AngleLinear(22, 11).to(device)


    output = torch.unsqueeze(torch.cat((output[0], output[1]), 0), 0)
    output = angl(output)
    f1, f2 = torch.squeeze(output[0].data), torch.squeeze(output[1].data)
    cosdistance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)

    D = torch.norm(x - x_adv)

    C = 0 if cosdistance < best_thresh else float('inf')
    return D + C

def generate_adversarial_face(net, x, T):
    # input
    #    L : the attack objective function
    #    x : torch tensor img with shape (1,3,112,96)
    # output
    #    : adversarial face image x
    _, _, H, W = x.shape
    m = 1 * 2 * 50
    k = m // 5
    C = torch.eye(m)
    p_c = torch.zeros(m)

    c_c = 0.01
    c_cov = 0.001
    sigma = 0.01
    success_rate = 0
    mu = 1
    x_adv = torch.randn_like(x)
    criterion(net, x, x_adv)

    for t in (range((T))):
        # x, x_adv = x.to('cpu'), x_adv.to('cpu')
        z = MultivariateNormal(loc=torch.zeros([m]), covariance_matrix=(sigma ** 2) * C).rsample()
        z = z.to(device)#
        # z = np.random.normal(loc=0.0, scale=(sigma**2) * C)

        zeroIdx = np.argsort(-C.diagonal())[k:]
        z[zeroIdx] = 0

        z = z.reshape([1, 1, 2, 50])
        z_ = F.interpolate(z, (H, W), mode='bilinear')
        z_ = z_.to(device)#
        z_ = z_ + mu * (x - x_adv)
        L_after = criterion(net, x, x_adv + z_)
        L_before = criterion(net, x, x_adv)

        if L_after < L_before:
            x_adv = x_adv + z_
            x_adv = (x_adv - x_adv.min()) / (x_adv.max() - x_adv.min())
            p_c = p_c.to(device)
            p_c = (1 - c_c) * p_c + np.sqrt(2 * (2 - c_c)) * z.reshape(-1) / sigma
            p_c = p_c.to('cpu')
            C[range(m), range(m)] = (1 - c_cov) * C.diagonal() + c_cov * (p_c) ** 2
            # print(L_after)
            # saveImg(x_adv, 'iter_' + str(t))
            success_rate += 1

        if t % 10 == 0:
            mu = mu * np.exp(success_rate / 10 - 1 / 5)
            success_rate = 0

        # print(t)

    return x_adv

def main():
    import pickle as pk
    import itertools
    xd = pk.load(open('../data/RML2016/RML2016.10a_dict.pkl', 'rb'), encoding='latin1')
    snrs, mods = list(map(lambda j: sorted(list(set(map(lambda x: x[j], xd.keys())))), [1, 0]))
    x = []
    y = []
    lbl = []
    snr_ = snrs  # [-10:] # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    accs = [[], []]
    for snr in snrs:
        x = []
        lbl = []
        for mod in mods:
            x.append(xd[(mod, snr)])
            for i in range(xd[(mod, snr)].shape[0]):
                lbl.append((mod, snr))
        x = np.vstack(x)
        length_x = x.shape[0]
        for i in range(x.shape[0]):
            x[i] = (x[i] - x[i].min()) / (x[i].max() - x[i].min())
        y = np.array([mods.index(i[0]) for i in lbl])
        test_dataset = rml2016a.IQ_dataset(x, y)
        test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

        victim_model = rml2016a.VTCNN2()
        victim_model.load_state_dict(torch.load("../saved_models/RML2016.pt"))
        victim_model.to(device)
        with torch.no_grad():
            acc_tar = 0
            acc_sub = 0
            for step, (x, y) in enumerate(test_dataloader):
                x, y = x.to(device), y.to(device)
                output = victim_model(x)
                acc_tar += torch.sum(torch.argmax(output, 1).eq(y)).data

                for i in tqdm(range(x.shape[0])):
                    x_adv = generate_adversarial_face(victim_model, torch.unsqueeze(x[i], 0), 100)
                    acc_sub += torch.sum(torch.argmax(victim_model(x_adv.to(device)), 1).eq(y[i]))
        print(torch.true_divide(acc_tar, length_x), torch.true_divide(acc_sub, length_x))
        accs[0].append(torch.true_divide(acc_tar, length_x))
        accs[1].append(torch.true_divide(acc_sub, length_x))
    plt.plot(snrs, accs[0], c='r', label='original data', alpha=0.7)
    plt.plot(snrs, accs[1], c='b', label="adversarial sample", alpha=0.7)
    plt.xlabel("SNR")
    plt.ylabel("ACC")
    plt.ylim(0, 1.0)
    plt.xticks(snrs)
    plt.yticks(np.arange(0, 1.001, 0.1))
    plt.grid(axis="y", ls='--')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()