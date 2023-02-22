import copy
import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from model import Net_cifar5
import torch.nn.functional as F
from load_dataset import load_cifar5_single_ex3, load_cifar5_un_fix_ex3


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


def test_grad(c, num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.manual_seed(1)
    path = f"./data_half_{c}"
    g = []
    train = load_cifar5_single_ex3(path, c, num)
    train_batch_size = num
    train_loader = DataLoader(train, batch_size=train_batch_size)
    model = Net_cifar5()
    model_path = f"./models_ex3/cifar5_cnn_{c}_{num}.pth"
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
    with open("grad_ex3.txt", "a+") as f:
        print(f"test:{g}", file=f)


def unlearning_grad(c, num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.manual_seed(1)
    path = f"./data_half_{c}"
    g = []
    train = load_cifar5_un_fix_ex3(path, c, num)
    train_batch_size = num
    train_loader = DataLoader(train, batch_size=train_batch_size)
    model = Net_cifar5()
    model_path = f"./models_ex3/cifar5_cnn_{c}_{num}.pth"
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
    with open("grad_ex3.txt", "a+") as f:
        print(f"unlearning:{g}", file=f)


def fake_test_grad(c, num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.manual_seed(1)
    path = f"./data_half_{c}"
    g = []
    train = load_cifar5_single_ex3(path, c, num)
    train_batch_size = num
    train_loader = DataLoader(train, batch_size=train_batch_size)
    model = Net_cifar5()
    model_path = f"./models_ex3/cifar5_cnn_{c}_{num}_fake.pth"
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
    with open("grad_ex3.txt", "a+") as f:
        print(f"fake_test:{g}", file=f)


def fake_unlearning_grad(c, num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.manual_seed(1)
    path = f"./data_half_{c}"
    g = []
    train = load_cifar5_un_fix_ex3(path, c, num)
    train_batch_size = num
    train_loader = DataLoader(train, batch_size=train_batch_size)
    model = Net_cifar5()
    model_path = f"./models_ex3/cifar5_cnn_{c}_{num}_fake.pth"
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
    with open("grad_ex3.txt", "a+") as f:
        print(f"fake_unlearning:{g}", file=f)


if __name__ == '__main__':
    un_list = [200, 400, 600, 800, 1000]
    for i in range(5):
        with open("grad_ex3.txt", "a+") as fi:
            print(f"-----------------{i}------------------", file=fi)
        test_grad(i, un_list[i])
        unlearning_grad(i, un_list[i])
        fake_test_grad(i, un_list[i])
        fake_unlearning_grad(i, un_list[i])
