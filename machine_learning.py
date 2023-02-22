import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from model import Net_cifar5
import torch.nn.functional as F
from load_dataset import load_cifar5


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


def test_model(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.manual_seed(1)
    path = "./data_half"
    train, test = load_cifar5(path)
    train_batch_size = 64
    test_batch_size = 64
    train_loader = DataLoader(train, batch_size=train_batch_size)
    test_loader = DataLoader(test, batch_size=test_batch_size)
    model = Net_cifar5().to(device)
    lr = 1.0
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    epochs = 50
    for epoch in range(1, epochs + 1):
        train_model(model, device, train_loader, optimizer, epoch)
        test_model(model, device, test_loader)
        scheduler.step()
    torch.save(model.state_dict(), "cifar5_cnn.pth")


if __name__ == '__main__':
    main()
