import torch
import torch.nn.functional as F
import torchvision
import rml2016a
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method as fgsm_cle
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent as pgd_cle

plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False 
plt.rcParams['savefig.dpi'] = 200 
plt.rcParams['figure.dpi'] = 200 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def PGD(model, X, y, criterion, iters=50, is_target=False, eps=0.5):
    x_adv = X
    x_org = X
    x_adv.to(device)
    x_org.to(device)

    for i in range(iters):
        x_adv.requires_grad = True
        output = model(x_adv)
        yy = y.add(2) % 10
        loss = criterion(output, yy)

        model.zero_grad()
        loss.backward()
        x_grad_sign = x_adv.grad.data.sign()
        x_adv = x_adv - eps * x_grad_sign
        eta = torch.clamp(x_adv - x_org, -eps, eps)
        x_adv = torch.clamp(x_org + eta, 0, 1).detach_()

    return x_adv, model(x_adv)


def criterion(net, x, x_adv):
    from torch.autograd import Variable
    # x, x_adv : torch.Tensor with shape [1,3,112,96]
    best_thresh = 0.3

    imglist = [x, x_adv]
    img = torch.vstack(imglist)
    img = Variable(img.float(), volatile=True).cuda()
    output = net(img)
    f1, f2 = output.data
    cosdistance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)

    D = torch.norm(x - x_adv)

    C = 0 if cosdistance < best_thresh else float('inf')
    return D + C

def generate_adversarial_face(net, x, T):
    from torch.distributions.multivariate_normal import MultivariateNormal
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

    for t in range(T):
        z = MultivariateNormal(loc=torch.zeros([m]), covariance_matrix=(sigma ** 2) * C).rsample()
        # z = np.random.normal(loc=0.0, scale=(sigma**2) * C)

        zeroIdx = np.argsort(-C.diagonal())[k:]
        z[zeroIdx] = 0

        z = z.reshape([1, 3, 45, 45])
        z_ = F.interpolate(z, (H, W), mode='bilinear')
        z_ = z_ + mu * (x - x_adv)
        L_after = criterion(net, x, x_adv + z_)
        L_before = criterion(net, x, x_adv)

        if L_after < L_before:
            x_adv = x_adv + z_
            p_c = (1 - c_c) * p_c + np.sqrt(2 * (2 - c_c)) * z.reshape(-1) / sigma
            C[range(m), range(m)] = (1 - c_cov) * C.diagonal() + c_cov * (p_c) ** 2
            print(L_after)
            # saveImg(x_adv, 'iter_' + str(t))
            success_rate += 1

        if t % 10 == 0:
            mu = mu * np.exp(success_rate / 10 - 1 / 5)
            success_rate = 0

        print(t)

    return x_adv


def model_output_plot(data, label_true, label_pred, nrows, ncols, title=None):

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(8,8))
    axes = ax.flatten()
    for i in range(nrows*ncols):
        axes[i].imshow(data[i])
        axes[i].set_title("{} -> {}".format(label_true[i], label_pred[i]))
    if title is not None:
        plt.suptitle(title)
    plt.show()

def load_models():
    victim_model = rml2016a.VTCNN2()
    victim_model.load_state_dict(torch.load("../saved_models/RML2016.pt"))
    victim_model.to(device)
    subsititude_model = rml2016a.VTCNN2()

    return victim_model, subsititude_model

def dtba(model, eps_list, save_path):
    _, _, _, testloader = rml2016a.IQ_data()
    data_size = len(testloader.dataset.indices)
    success_rate = []
    transferability = []
    true_fl = []
    num_true_f1 = []
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    adversarial_data = {}
    for eps in eps_list:
        data_matrix = []
        # models
        victim_model, _ = load_models()
        subsititude_model = model
        acc_vic = 0
        acc_sub = 0
        acc_adv_vic = 0
        acc_adv_sub = 0
        true_fl_ = 0
        lem_true_fl = 0
        len_tran = 0
        len_x = 0
        for step, (x, y) in enumerate(tqdm(testloader)):
            len_x += x.shape[0]
            x, y = x.to(device), y.to(device)

            output_vic, _, acc = rml2016a.test(victim_model, x, y, criterion)
            acc_vic += acc
            output_sub, _, acc = rml2016a.test(subsititude_model, x, y, criterion)
            acc_sub += acc


            x_adv = fgsm_cle(subsititude_model, x, eps, np.inf)
            # x_adv = pgd_cle(subsititude_model, x, eps, eps, 50, norm=np.inf)
            output_adv_vic, _, acc = rml2016a.test(victim_model, x_adv, y, criterion)
            acc_adv_vic += acc
            output_adv_sub, _, acc = rml2016a.test(subsititude_model, x_adv, y, criterion)
            acc_adv_sub += acc

            suc_idx_vic = torch.where(torch.argmax(output_adv_vic, 1).ne(y))[0]
            wro_idx_vic = torch.where(torch.argmax(output_vic, 1).ne(y))[0]
            true_suc_vic = set(suc_idx_vic.cpu().numpy()) - set(wro_idx_vic.cpu().numpy()) # transfer ability
            suc_idx_sub = torch.where(torch.argmax(output_adv_sub, 1).ne(y))[0]
            wro_idx_sub = torch.where(torch.argmax(output_sub, 1).ne(y))[0]
            true_suc_sub = set(suc_idx_sub.cpu().numpy()) - set(wro_idx_sub.cpu().numpy()) # success rate

            len_tran += len(true_suc_vic)
            lem_true_fl += len(true_suc_sub)
            true_fl_ += len(true_suc_vic & true_suc_sub)

            data_matrix.append(x_adv.detach().cpu().numpy())
        data_matrix = np.vstack(data_matrix)
        adversarial_data[f'{eps}'] = data_matrix




        acc_vic = torch.true_divide(acc_vic, data_size)
        acc_sub = torch.true_divide(acc_sub, data_size)
        acc_adv_vic = torch.true_divide(acc_adv_vic, data_size)
        acc_adv_sub = torch.true_divide(acc_adv_sub, data_size)
        # success_rate.append(((acc_sub - acc_adv_sub)/acc_sub).cpu().item())
        # transferability.append(((acc_vic - acc_adv_vic)/acc_vic).cpu().item())

        success_rate.append((torch.true_divide(lem_true_fl, len_x * acc_sub).cpu().item()))
        transferability.append(torch.true_divide(len_tran, len_x).cpu().item())

        if lem_true_fl == 0:
            true_fl.append(0)
        else:
            true_fl.append(torch.true_divide(true_fl_, lem_true_fl).cpu().item())
        num_true_f1.append(lem_true_fl)
        print(acc_vic, acc_sub, torch.true_divide(true_fl_, lem_true_fl))
    # np.save(save_path, adversarial_data)
    return acc_vic, acc_sub, success_rate, transferability, true_fl, num_true_f1


if __name__ == "__main__":
    print(dtba())
