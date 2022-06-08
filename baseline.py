import os
import numpy as np
import glob
import PIL.Image as Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Check if torch available
if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import wandb
from omegaconf import OmegaConf

DIR= os.getcwd()
config = OmegaConf.load(f"{DIR}/config/project1.yaml")

wandb.init(project="Project1", entity="dlincvg1", config=dict(config),)

# load configs
_num_epoch = config.epoch
batch_size = config.batch_size
learning_rate = config.learning_rate


#%% Dataloader

class Hotdog_NotHotdog(torch.utils.data.Dataset):
    def __init__(self, train, transform, data_path='/dtu/datasets1/02514/hotdog_nothotdog'):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, 'train' if train else 'test')
        image_classes = [os.path.split(d)[1] for d in glob.glob(data_path +'/*') if os.path.isdir(d)]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + '/*/*.jpg')
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image)
        return X, y

size = 128
train_transform = transforms.Compose([transforms.Resize((size, size)), 
                                    transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((size, size)), 
                                    transforms.ToTensor()])

trainset = Hotdog_NotHotdog(train=True, transform=train_transform)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
testset = Hotdog_NotHotdog(train=False, transform=test_transform)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)


#%% Network

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.convolutional = nn.Sequential(
                nn.Conv2d(in_channels=3,
                        out_channels=8,
                        kernel_size=(3,3),
                        padding='same'),
                nn.ReLU(),
                nn.Conv2d(in_channels=8,
                          out_channels=8,
                          kernel_size=(3,3),
                         padding='same'),
                nn.ReLU(),
                nn.Conv2d(in_channels=8,
                          out_channels=16,
                          kernel_size=(3,3),
                         padding='same'),
        )

        self.fully_connected = nn.Sequential(
                nn.Linear(128*128*16, 500),
                nn.ReLU(),
                nn.Linear(500, 2),
        )
    
    def forward(self, x):
        x = self.convolutional(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x


model = Network()
model.to(device)
wandb.watch(model, log_freq=100)


       
#%% train

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

num_epochs = _num_epoch

for epoch in tqdm(range(num_epochs), unit='epoch'):
    #For each epoch
    train_correct = 0
    model.train()
    for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.to(device), target.to(device)
        #Zero the gradients computed for each weight
        optimizer.zero_grad()
        #Forward pass your image through the network
        output = model(data)
        #Compute the loss
        loss_train = criterion(output, target)
        #Backward pass through the network
        loss_train.backward()
        #Update the weights
        optimizer.step()
        
        #Compute how many were correctly classified
        predicted = output.argmax(1)
        train_correct += (target==predicted).sum().cpu().item()
    #Comput the test accuracy
    test_correct = 0
    
    with torch.no_grad():
        model.eval()
        for data, target in test_loader:
            
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)
            loss_test = criterion(output, target)

            predicted = output.argmax(1)
            
            test_correct += (target==predicted).sum().item()
        train_acc = train_correct/len(trainset)
        test_acc = test_correct/len(testset)
        
    print("\n \n Epoch: {epoch}/{total} \n Accuracy train: {train:.1f}%\t test: {test:.1f}% \n Loss: \t train: {loss_train:.3f} \t test: {loss_test:.3f}\n".format(
                                                        epoch=epoch, total=_num_epoch,
                                                        test=100*test_acc, train=100*train_acc,
                                                        loss_train=loss_train, loss_test=loss_test))
    
    
    wandb.log({"train_acc":train_acc})
    wandb.log({"test_acc":test_acc})

    wandb.log({"train_loss":loss_train})
    wandb.log({"test_loss":loss_test})