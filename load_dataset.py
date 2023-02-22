import random
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import transforms


def load_cifar5(path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train_dataset = datasets.ImageFolder(root=f"{path}/train", transform=transform)
    n = len(train_dataset)
    n_train = random.sample(range(n), n)
    train_dataset = Subset(train_dataset, n_train)
    test_dataset = datasets.ImageFolder(root=f"{path}/test", transform=transform)
    m = len(test_dataset)
    m_test = random.sample(range(m), m)
    test_dataset = Subset(test_dataset, m_test)
    return train_dataset, test_dataset


def load_cifar5_un(path, un_c, un_num):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train_dataset = datasets.ImageFolder(root=f"{path}/train", transform=transform)
    n = len(train_dataset)
    n_train = set(range(n))
    un = random.sample(range(5000 * un_c, 5000 * un_c + 5000), un_num)
    n_train = n_train - set(un)
    n_train = list(n_train)
    n_train = random.sample(n_train, len(n_train))
    train_dataset = Subset(train_dataset, n_train)
    test_dataset = datasets.ImageFolder(root=f"{path}/test", transform=transform)
    m = len(test_dataset)
    m_test = random.sample(range(m), m)
    test_dataset = Subset(test_dataset, m_test)
    return train_dataset, test_dataset


def load_cifar5_single(path, c):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_dataset = datasets.ImageFolder(root=f"{path}/test", transform=transform)
    m_test = random.sample(range(1000 * c, 1000 * c + 1000), 1000)
    test_dataset = Subset(test_dataset, m_test)
    return test_dataset


def load_cifar5_un_fix(path, un_c, un_num):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train_dataset = datasets.ImageFolder(root=f"{path}/train", transform=transform)
    n = len(train_dataset)
    n_train = set(range(n))

    un = range(5000 * un_c, 5000 * un_c + un_num)
    n_train = n_train - set(un)
    n_train = list(n_train)
    n_train = random.sample(n_train, len(n_train))
    train_dataset = Subset(train_dataset, n_train)
    test_dataset = datasets.ImageFolder(root=f"{path}/test", transform=transform)
    m = len(test_dataset)
    m_test = random.sample(range(m), m)
    test_dataset = Subset(test_dataset, m_test)
    return train_dataset, test_dataset


def load_cifar5_un_fix_fake(path, un_c, un_num):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train_dataset = datasets.ImageFolder(root=f"{path}/train", transform=transform)
    n = len(train_dataset)
    n_train = set(range(n))

    un = range(5000 * un_c + un_num, 5000 * un_c + un_num * 2)
    n_train = n_train - set(un)
    n_train = list(n_train)
    n_train = random.sample(n_train, len(n_train))
    train_dataset = Subset(train_dataset, n_train)
    test_dataset = datasets.ImageFolder(root=f"{path}/test", transform=transform)
    m = len(test_dataset)
    m_test = random.sample(range(m), m)
    test_dataset = Subset(test_dataset, m_test)
    return train_dataset, test_dataset


def load_cifar5_single_ex3(path, c, num):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_dataset = datasets.ImageFolder(root=f"{path}/test", transform=transform)
    m_test = range(1000 * c, 1000 * c + num)
    test_dataset = Subset(test_dataset, m_test)
    return test_dataset


def load_cifar5_un_fix_ex3(path, c, num):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train_dataset = datasets.ImageFolder(root=f"{path}/train", transform=transform)
    un = range(5000 * c, 5000 * c + num)
    train_dataset = Subset(train_dataset, un)
    return train_dataset


if __name__ == '__main__':
    pass
