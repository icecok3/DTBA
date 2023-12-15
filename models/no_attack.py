import torch
import torch.nn as nn
import rml2016a
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def labeling(model, dataloader):
    y_res = None
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_vic = model(x)
            if y_res != None:
                y_res = torch.cat([y_res, y_vic], dim=0)
            else:
                y_res = y_vic
    y_res = torch.argmax(y_res, dim=1)
    dataset_vim = rml2016a.IQ_dataset(dataloader.dataset.dataset.data[dataloader.dataset.indices], y_res)
    dataloader_vim = DataLoader(dataset_vim, batch_size=256, num_workers=0)

    return dataloader_vim

def training(model, dataloader):
    epoches = 50
    lr = 0.001
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    min_loss = 1e10
    print(len(dataloader.dataset))
    for epoch in range(epoches):
        loss_sum = 0
        acc_num = 0
        for i, (x, y) in enumerate(tqdm(dataloader)):
            x, y = x.to(device), y.to(device)
            output, loss, acc = rml2016a.train(model, x, y, criterion, optimizer)
            loss_sum += loss
            acc_num += acc
        if epoch % 1 == 0:
            print("="*50)
            print("Epoch:{}, loss:{}, acc:{}".format(epoch,
                                                      torch.true_divide(loss_sum, len(dataloader.dataset)),
                                                      torch.true_divide(acc_num, len(dataloader.dataset)),
                                                      ))
        if torch.true_divide(acc_num, len(dataloader.dataset)) < min_loss:
            torch.save(model.state_dict(), '../saved_models/RML2016_contrast.pt')

    return model

def main():
    victim_model = rml2016a.VTCNN2()
    victim_model.load_state_dict(torch.load("../saved_models/RML2016.pt"))
    victim_model.to(device)
    sub_model = rml2016a.VTCNN2()
    sub_model.to(device)

    _, _, trainloader, testloader = rml2016a.IQ_data()
    vic_dataloader = labeling(victim_model, testloader)
    sub_model = training(sub_model, vic_dataloader)


if __name__ == "__main__":
    main()

