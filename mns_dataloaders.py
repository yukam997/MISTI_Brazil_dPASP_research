import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import random

NUM_RANGE = (0,9)
# Download training data from open datasets.
training_data_atoms = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data_atoms = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# separate images by digits
test_digits= [[] for i in range(10)]
for img, label in test_data_atoms:
    test_digits[label]+=img
train_digits= [[] for i in range(10)]
for img, label in training_data_atoms:
    train_digits[label]+=img

def convert_dataset(training_data,test_data):
    return DataLoader(training_data, batch_size=64, shuffle=True),DataLoader(test_data, batch_size=64, shuffle=True)

def generate_num_set(n,num_digits = 3):
    small = max(n-NUM_RANGE[1],0)
    big = min(NUM_RANGE[1],n - small)
    num1 = random.randint(small,big)
    num2 = random.randint(small,big)
    num3 = random.randint(small,big)
    if num_digits == 5:
        return (num1,n-num1,num2,n-num2,num3,n-num3)
    return (num1,n-num1,num2,n-num2)

def five_channels(train_size=100000,test_size=1000):
    def generate_train_data_point(numbers):
        data_point = [random.choice(train_digits[numbers[i]])[None,:,:] for i in range(5)]
        return (torch.cat(data_point),numbers[5])

    def generate_test_data_point(numbers):
        data_point = [random.choice(test_digits[numbers[i]])[None,:,:] for i in range(5)]
        return (torch.cat(data_point),numbers[5])
    
    training_data=[]
    for i in range(train_size):
        total = random.randint(4,16)
        num = generate_num_set(total,5)
        training_data += [generate_train_data_point(num)]
        
    test_data=[]
    for i in range(test_size):
        total = random.randint(4,16)
        num = generate_num_set(total,5)
        test_data += [generate_test_data_point(num)]
        
    return convert_dataset(training_data,test_data)


def five_concat(train_size=100000,test_size=1000):
    def generate_train_data_point(numbers):
        data_point = [random.choice(train_digits[numbers[i]])[None,:,:] for i in range(5)]
        return (torch.cat(data_point),numbers[5])

    def generate_test_data_point(numbers):
        data_point = [random.choice(test_digits[numbers[i]])[None,:,:] for i in range(5)]
        return (torch.cat(data_point),numbers[5])
    
    training_data=[]
    for i in range(train_size):
        total = random.randint(4,16)
        num = generate_num_set(total,5)
        training_data += [generate_train_data_point(num)]
        
    test_data=[]
    for i in range(test_size):
        total = random.randint(4,16)
        num = generate_num_set(total,5)
        test_data += [generate_test_data_point(num)]
    return convert_dataset(training_data,test_data)
    

def two_concat(train_size=100000,test_size=1000):
    def generate_train_data_point(numbers):
        im1 = torch.cat([random.choice(train_digits[numbers[0]]),torch.zeros(10, 28),random.choice(train_digits[numbers[1]])])
        im2 = torch.cat([random.choice(train_digits[numbers[2]]),torch.zeros(10, 28),random.choice(train_digits[numbers[3]])])
        data_point = [im1[None,:,:],im2[None,:,:]]
        return (torch.cat(data_point),torch.tensor(numbers))

    def generate_test_data_point(numbers):
        im1 = torch.cat([random.choice(test_digits[numbers[0]]),torch.zeros(8, 28),random.choice(test_digits[numbers[1]])])
        im2 = torch.cat([random.choice(test_digits[numbers[2]]),torch.zeros(8, 28),random.choice(test_digits[numbers[3]])])
        data_point = [im1[None,:,:],im2[None,:,:]]
        return (torch.cat(data_point),torch.tensor(numbers))


    training_data=[]
    for _ in range(train_size):
        total = random.randint(4,16)
        num = generate_num_set(total)
        training_data += [generate_train_data_point(num)]
        
    test_data=[]
    for _ in range(test_size):
        total = random.randint(4,16)
        num = generate_num_set(total)
        test_data += [generate_test_data_point(num)]
    return convert_dataset(training_data,test_data)


def three_channels(train_size=100000,test_size=1000):
    def generate_train_data_point(numbers):
        data_point = [random.choice(train_digits[numbers[i]])[None,:,:] for i in range(3)]
        return (torch.cat(data_point),numbers[3])

    def generate_test_data_point(numbers):
        data_point =[random.choice(test_digits[numbers[i]])[None,:,:] for i in range(3)]
        return (torch.cat(data_point),numbers[3])

    training_data=[]
    for i in range(train_size):
        total = random.randint(4,16)
        num = generate_num_set(total)
        training_data += [generate_train_data_point(num)]
        
    test_data=[]
    for i in range(test_size):
        total = random.randint(4,16)
        num = generate_num_set(total)
        test_data += [generate_test_data_point(num)]
    return convert_dataset(training_data,test_data)