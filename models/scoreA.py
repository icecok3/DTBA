import BBA_iq
import AD_iq
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


def score(random_size, attack_name):
    import pickle as pk
    import itertools
    xd = pk.load(open('../data/RML2016/RML2016.10a_dict.pkl', 'rb'), encoding='latin1')
    snrs, mods = list(map(lambda j: sorted(list(set(map(lambda x: x[j], xd.keys())))), [1, 0]))

    lbl = []
    snrs_ = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]  # [-10:] # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    x = []
    y = []
    for snr in snrs:

        for m in mods:
            x_ = xd[(m, snr)]
            np.random.seed(0)
            x_ = x_[np.random.choice(np.arange(0, x_.shape[0]), random_size, replace=False)]
            x.append(x_)
            y.append([mods.index(m)] * random_size)
    x = np.vstack(x)
    for i in range(x.shape[0]):
        x[i] = (x[i] - x[i].min()) / (x[i].max() - x[i].min())
    y = np.hstack(y)
    print(x.shape, y.shape)
    testdata = rml2016a.IQ_dataset(x, y)
    testloader = DataLoader(testdata, batch_size=256, shuffle=False, num_workers=0)


    ##### FGSM #####
    for ei, eps in enumerate([0.1, 0.2]):
        adversarial_query = 0
        dis_size = 0
        y_vic_num_ = 0
        query_num = 20460
    
        for step, (x, y) in enumerate(testloader):
            x, y = x.to(device), y.to(device)
            model = rml2016a.VTCNN2()
            model.load_state_dict(torch.load('../saved_models/RML2016.pt'))
            model.to(device)
            model.eval()
    
            criterion = torch.nn.CrossEntropyLoss()
            y_a = torch.zeros_like(y)
            attack_model = rml2016a.VTCNN2()
            attack_model.load_state_dict(torch.load('../saved_models/SUB_VTCNN2_jr0.2_rr3.pt'))
            attack_model.to(device)
    
            x_adv = fgsm_cle(attack_model, x, eps, np.inf)
            y_adv = torch.argmax(attack_model(x_adv), 1)
    
            suc_idx_att = torch.where(y_adv.ne(y) == True)[0]
            x_adv = x_adv[suc_idx_att]
    
            y_vic = model(x_adv)
            y_vic_num = torch.where(torch.argmax(y_vic, 1).ne(y[suc_idx_att]) == True)[0]
            y_vic_num_ += y_vic_num.shape[0]
    
            adversarial_query += x_adv.shape[0]
    
            for i in range(x_adv.shape[0]):
                dis_size += F.mse_loss(x_adv[i], x[suc_idx_att[i]])
    
        dis_size = dis_size / y_vic_num_
        score  = y_vic_num_ / (dis_size * (adversarial_query + query_num))
    
        res = {'adv_query':adversarial_query, 'succ_num':y_vic_num_, 'dis_size':dis_size, 'score':score}
        print(res)
    
    #### PGD #####
    for ei, eps in enumerate([0.1, 0.2]):
        adversarial_query = 0
        dis_size = 0
        y_vic_num_ = 0
        query_num = 20460
    
        for step, (x, y) in enumerate(testloader):
            x, y = x.to(device), y.to(device)
            model = rml2016a.VTCNN2()
            model.load_state_dict(torch.load('../saved_models/RML2016.pt'))
            model.to(device)
            model.eval()
    
            criterion = torch.nn.CrossEntropyLoss()
            y_a = torch.zeros_like(y)
            attack_model = rml2016a.VTCNN2()
            attack_model.load_state_dict(torch.load('../saved_models/SUB_VTCNN2_jr0.2_rr3.pt'))
            attack_model.to(device)
    
            x_adv = pgd_cle(attack_model, x, eps, eps, 50, norm=np.inf)
            y_adv = torch.argmax(attack_model(x_adv), 1)
    
            suc_idx_att = torch.where(y_adv.ne(y) == True)[0]
            x_adv = x_adv[suc_idx_att]
    
            y_vic = model(x_adv)
            y_vic_num = torch.where(torch.argmax(y_vic, 1).ne(y[suc_idx_att]) == True)[0]
            y_vic_num_ += y_vic_num.shape[0]
    
            adversarial_query += x_adv.shape[0]
            t = x_adv.shape[0]
    
            for i in y_vic_num:
                dis_size += F.mse_loss(x_adv[i], x[suc_idx_att][i])
        dis_size = dis_size / y_vic_num_
        score  = y_vic_num_ / (dis_size * (adversarial_query + query_num))
    
        res = {'adv_query':adversarial_query, 'succ_num':y_vic_num_, 'dis_size':dis_size, 'score':score}
        print(res)

    #### evalution attack #####
    for ei, eps in enumerate([75, 100]):
        adversarial_query = x.shape[0]
        dis_size = 0
        y_vic_num_ = 0
        query_num = 2*eps + 1
    
        model = rml2016a.VTCNN2()
        model.load_state_dict(torch.load('../saved_models/RML2016.pt'))
        model.to(device)
        model.eval()
    
        attack_model = rml2016a.VTCNN2()
        attack_model.load_state_dict(torch.load('../saved_models/SUB_VTCNN2_jr0.2_rr3.pt'))
        attack_model.to(device)
    
        for i in tqdm(range((x.shape[0]))):
            xx = torch.unsqueeze(torch.tensor(x[i]), 0)
            xx = xx.to(device)
            x_adv = generate_adversarial_face(model, torch.unsqueeze(xx, 0), eps)
            x_adv = (x_adv - x_adv.min()) / (x_adv.max() - x_adv.min())
            y_adv = torch.argmax(model(x_adv), 1).eq(y[i])
            if not y_adv:
                y_vic_num_ += 1
                dis_size += F.mse_loss(x_adv, xx)
    
        dis_size = dis_size / y_vic_num_
        score  = y_vic_num_ / (dis_size * (adversarial_query * query_num))
    
        res = {'adv_query':adversarial_query, 'succ_num':y_vic_num_, 'dis_size':dis_size, 'score':score}
        print(res)



if __name__ == '__main__':
    score(10)