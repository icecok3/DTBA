import torch
import torch.nn as nn
import torch.nn.functional as F
import rml2016a
import copy
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from EDBA import generate_adversarial_face

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method as fgsm_cle
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent as pgd_cle

plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False 
plt.rcParams['savefig.dpi'] = 200 
plt.rcParams['figure.dpi'] = 1000 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_signal():
    import pickle as pk
    import itertools

    xd = pk.load(open('../data/RML2016/RML2016.10a_dict.pkl', 'rb'), encoding='latin1')
    snrs, mods = list(map(lambda j: sorted(list(set(map(lambda x: x[j], xd.keys())))), [1, 0]))
    x = []
    y = []
    lbl = []
    snr_ = snrs 

    for snr in [10]:
        for mod in mods:
            x.append(xd[(mod, snr)])
            for i in range(xd[(mod, snr)].shape[0]):
                lbl.append((mod, snr))
    x = np.vstack(x)
    y = np.array([mods.index(i[0]) for i in lbl])
    print(x.shape, y.shape)
    for i in range(x.shape[0]):
        x[i] = (x[i] - x[i].min()) / (x[i].max() - x[i].min())
    testdata = rml2016a.IQ_dataset(x, y)
    testloader = DataLoader(testdata, batch_size=1000, shuffle=False, num_workers=0)

    x, y = (next(iter(testloader)))
    x, y = x.to(device), y.to(device)
    x = x.to(device)
    x = x[256]
    y = torch.tensor([y[256]])
    y = y.to(device)

    idex1 = 100
    index2 = 1100

    fig, axes = plt.subplots(nrows=3, ncols=2,  figsize=(8,8))
    axes = axes.flatten()

    model = rml2016a.VTCNN2()
    model.load_state_dict(torch.load('../saved_models/RML2016.pt'))
    model.to(device)
    model.eval()

    attack_model = rml2016a.VTCNN2()
    attack_model.load_state_dict(torch.load('../saved_models/SUB_VTCNN2_jr0.2_rr3.pt'))
    attack_model.to(device)
    attack_model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    x_adv = fgsm_cle(model, torch.unsqueeze(x, 0), 0.1, np.inf)
    x_adv = torch.squeeze(torch.squeeze(x_adv))
    x_ori = copy.deepcopy(x)
    x_ori = torch.squeeze(torch.squeeze(x_ori))

    mseloss = F.mse_loss(x_ori, x_adv).item()

    com_iq = x_ori[0, :] + 1j * x_ori[1, :]
    com_iq_adv = x_adv[0, :] + 1j * x_adv[1, :]
    axes[0].plot(np.abs(com_iq.detach().cpu().numpy()), c='b', label="original data", alpha=0.7)
    axes[0].plot(np.abs(com_iq_adv.detach().cpu().numpy()), c='r', label="adversarial sample", alpha=0.7)
    axes[0].set_xlabel(r'FGSM $ \varepsilon = 0.1$ on target model $M_{T}$ ' + '\n' + f'MSE : {round(mseloss, 5)}', fontdict={'size': 10})

    x_adv = fgsm_cle(attack_model, torch.unsqueeze(x, 0), 0.1, np.inf)
    x_adv = torch.squeeze(torch.squeeze(x_adv))
    x_ori = copy.deepcopy(x)
    x_ori = torch.squeeze(torch.squeeze(x_ori))

    mseloss = F.mse_loss(x_ori, x_adv).item()

    com_iq = x_ori[0, :] + 1j * x_ori[1, :]
    com_iq_adv = x_adv[0, :] + 1j * x_adv[1, :]
    axes[1].plot(np.abs(com_iq.detach().cpu().numpy()), c='b', alpha=0.7)
    axes[1].plot(np.abs(com_iq_adv.detach().cpu().numpy()), c='r', alpha=0.7)
    axes[1].set_xlabel(r'FGSM $ \varepsilon = 0.1$ on local model $M_{L}$'  + '\n' + f'MSE : {round(mseloss, 5)}', fontdict={'size': 10})

    x_adv = pgd_cle(model, torch.unsqueeze(x, 0), 0.1, 0.1, 50, norm=np.inf)
    x_adv = torch.squeeze(torch.squeeze(x_adv))
    x_ori = copy.deepcopy(x)
    x_ori = torch.squeeze(torch.squeeze(x_ori))

    mseloss = F.mse_loss(x_ori, x_adv).item()

    com_iq = x_ori[0, :] + 1j * x_ori[1, :]
    com_iq_adv = x_adv[0, :] + 1j * x_adv[1, :]
    axes[2].plot(np.abs(com_iq.detach().cpu().numpy()), c='b', alpha=0.7)
    axes[2].plot(np.abs(com_iq_adv.detach().cpu().numpy()), c='r', alpha=0.7)
    axes[2].set_xlabel(r'PGD $ \varepsilon = 0.1$ on target model $M_{T}$'  + '\n' + f'MSE : {round(mseloss, 5)}', fontdict={'size': 10})

    x_adv = pgd_cle(attack_model, torch.unsqueeze(x, 0), 0.1, 0.1, 50, norm=np.inf)
    x_adv = torch.squeeze(torch.squeeze(x_adv))
    x_ori = copy.deepcopy(x)
    x_ori = torch.squeeze(torch.squeeze(x_ori))

    mseloss = F.mse_loss(x_ori, x_adv).item()

    com_iq = x_ori[0, :] + 1j * x_ori[1, :]
    com_iq_adv = x_adv[0, :] + 1j * x_adv[1, :]
    axes[3].plot(np.abs(com_iq.detach().cpu().numpy()), c='b', alpha=0.7)
    axes[3].plot(np.abs(com_iq_adv.detach().cpu().numpy()), c='r', alpha=0.7)
    axes[3].set_xlabel(r'PGD $ \varepsilon = 0.1$ on local model $M_{L}$'  + '\n' + f'MSE : {round(mseloss, 5)}', fontdict={'size': 10})

    x_adv = generate_adversarial_face(model, torch.unsqueeze(x, 0), 75)
    x_adv = torch.squeeze(torch.squeeze(x_adv))

    x_ori = copy.deepcopy(x)
    x_ori = torch.squeeze(torch.squeeze(x_ori))

    mseloss = F.mse_loss(x_ori, x_adv).item()

    com_iq = x_ori[0, :] + 1j * x_ori[1, :]
    com_iq_adv = x_adv[0, :] + 1j * x_adv[1, :]
    axes[4].plot(np.abs(com_iq.detach().cpu().numpy()), c='b', alpha=0.7)
    axes[4].plot(np.abs(com_iq_adv.detach().cpu().numpy()), c='r', alpha=0.7)
    axes[4].set_xlabel(r'EDBA $ T = 75$'  + '\n' + f'MSE : {round(mseloss, 5)}', fontdict={'size': 10})

    x_adv = generate_adversarial_face(model, torch.unsqueeze(x, 0), 100)
    x_adv = torch.squeeze(torch.squeeze(x_adv))
    x_adv = (x_adv - x_adv.min()) / (x_adv.max() - x_adv.min())
    x_ori = copy.deepcopy(x)
    x_ori = torch.squeeze(torch.squeeze(x_ori))

    mseloss = F.mse_loss(x_ori, x_adv).item()

    com_iq = x_ori[0, :] + 1j * x_ori[1, :]
    com_iq_adv = x_adv[0, :] + 1j * x_adv[1, :]
    axes[5].plot(np.abs(com_iq.detach().cpu().numpy()), c='b', alpha=0.7)
    axes[5].plot(np.abs(com_iq_adv.detach().cpu().numpy()), c='r', alpha=0.7)
    axes[5].set_xlabel(r'EDBA $ T = 100$'  + '\n' + f'MSE : {round(mseloss, 5)}', fontdict={'size': 10})
    plt.subplots_adjust(bottom=0.15, wspace=0.2, hspace=0.5)
    leg = fig.legend(bbox_to_anchor=(0.5, 0), loc = 'lower center', fontsize=8.5, prop={'size': 10}, ncol=2)
    
    plt.savefig('../imagines/show_adversarial_examples.pdf')


if __name__ == '__main__':
    print_signal()