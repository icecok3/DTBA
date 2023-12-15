import torch
import torch.nn.functional as F
import torchvision
import BBA_iq
import AD_iq
import rml2016a
import copy
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False 
plt.rcParams['savefig.dpi'] = 200 
plt.rcParams['figure.dpi'] = 200 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    model_name = ['VTCNN2']
    models = [rml2016a.VTCNN2()]
    jabco_rate = [0.01,0.02,0.1,0.2]
    random_choose_rate = [1,3,5]
    eps_list = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]

    for m_idx, m in enumerate(model_name):
        print("{} : {}\n".format(m_idx, m))
        import itertools
        for jr, rcr in itertools.product(jabco_rate, random_choose_rate):
            print("\tjabco rate:{} | random choose rate:{}\n".format(jr, rcr))
            save_path = "../saved_models/top3_SUB_{}_jr{}_rr{}.pt".format(m, jr, rcr)
            query_num = BBA_iq.Black_Box_Attack(copy.deepcopy(models[m_idx]).to(device), 5, rcr, jr, save_path)
            torch.cuda.empty_cache()
            model =  copy.deepcopy(models[m_idx]).to(device)
            model.load_state_dict(torch.load(save_path))
            # model = copy.deepcopy(models[m_idx]).to(device).load_state_dict(torch.load(save_path))
            acc_vic, acc_sub, success_rate, transferability, true_fl, len_true_fl = AD_iq.adversial_attack(model,
                                                                                                           eps_list,
                                                                                                           '../data/adversarial_data/SUB_{}_jr{}_rr{}_mmn'.format(m, jr, rcr))
            result = {"query_num": query_num,
                      "acc_vic": acc_vic.cpu().numpy().item(),
                      "acc_sub": acc_sub.cpu().numpy().item(),
                      "succ": success_rate,
                      "tran": transferability,
                      "true_fl": true_fl,
                      "len_true_fl": len_true_fl
                      }

            import json
            result = json.dumps(result)
            with open('../results/top3_SUB_{}_jr{}_rr{}.json'.format(m, jr, rcr), 'a') as json_file:
                json_file.seek(0)
                json_file.truncate()
                json_file.write(result)
            del model
            torch.cuda.empty_cache()
    # model_save_path = ["../saved_models/SUB_{}.pt".format(x) for x in model_name]

if __name__ == '__main__':
    main()