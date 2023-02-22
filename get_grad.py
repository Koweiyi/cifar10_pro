import copy
import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from model import Net_cifar5
import torch.nn.functional as F
from load_dataset import load_cifar5_single


def train_model(model, device, train_loader, optimizer, epoch):
    model.train()
    print(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def get_grad(weights, lr=1.0):
    grad = weights[1].flatten() - weights[0].flatten()
    grad /= lr
    g = 0
    for item in grad:
        g += abs(item)
    return g


def local_grad():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.manual_seed(1)
    path = "./data"
    for c in range(5):
        train = load_cifar5_single(path, c)
        train_batch_size = 1000
        train_loader = DataLoader(train, batch_size=train_batch_size)
        model = Net_cifar5()
        g = []
        for num in range(1000, 5001, 1000):
            model_path = f"./local_models/cifar5_cnn_{c}_{num}.pth"
            model.load_state_dict(torch.load(model_path))
            model = model.to(device)
            lr = 1.0
            optimizer = optim.Adadelta(model.parameters(), lr=lr)
            epochs = 1
            w0 = copy.deepcopy(model.state_dict()["module.6.weight"])
            W = [w0]
            for epoch in range(1, epochs + 1):
                train_model(model, device, train_loader, optimizer, epoch)
                w_temp = copy.deepcopy(model.state_dict()["module.6.weight"])
                W.append(w_temp)
            grad = get_grad(W)
            g.append(grad)
        with open("grad_local.txt", "a+") as f:
            print(g, file=f)


def origin_grad(c):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.manual_seed(1)
    path = f"./data_half_{c}"
    g = []
    train = load_cifar5_single(path, c)
    train_batch_size = 1000
    train_loader = DataLoader(train, batch_size=train_batch_size)
    model = Net_cifar5()
    model_path = f"./models/cifar5_cnn.pth"
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    lr = 1.0
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    epochs = 1
    w0 = copy.deepcopy(model.state_dict()["module.6.weight"])
    W = [w0]
    for epoch in range(1, epochs + 1):
        train_model(model, device, train_loader, optimizer, epoch)
        w_temp = copy.deepcopy(model.state_dict()["module.6.weight"])
        W.append(w_temp)
    grad = get_grad(W)
    g.append(grad)
    with open("grad_attack.txt", "a+") as f:
        print(f"origin:{g}", file=f)


def attack_grad(c, num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.manual_seed(1)
    path = f"./data_half_{c}"
    g = []
    train = load_cifar5_single(path, c)
    train_batch_size = 1000
    train_loader = DataLoader(train, batch_size=train_batch_size)
    model = Net_cifar5()
    model_path = f"./models/cifar5_cnn_{c}_{num}.pth"
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    lr = 1.0
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    epochs = 1
    w0 = copy.deepcopy(model.state_dict()["module.6.weight"])
    W = [w0]
    for epoch in range(1, epochs + 1):
        train_model(model, device, train_loader, optimizer, epoch)
        w_temp = copy.deepcopy(model.state_dict()["module.6.weight"])
        W.append(w_temp)
    grad = get_grad(W)
    g.append(grad)
    with open("grad_attack.txt", "a+") as f:
        print(f"unlearning:{g}", file=f)


if __name__ == '__main__':
    local_grad()
    attack_list = [1000, 2000, 3000, 4000, 5000]
    for i in range(5):
        with open("grad_attack.txt", "a+") as fi:
            print(f"------------{i}-------------", file=fi)
        origin_grad(i)
        attack_grad(i, attack_list[i])

