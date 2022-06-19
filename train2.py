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
from albumentations.pytorch import ToTensorV2
import albumentations as A

import matplotlib.pyplot as plt

# Check if torch available
if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import wandb
from omegaconf import OmegaConf

DIR= os.path.dirname(os.path.realpath(__file__))
config = OmegaConf.load(f"{DIR}/config/config.yaml")

# Hyperparameter
EPOCHS = config.EPOCHS
BATCH_SIZE = config.BATCH_SIZE
LEARNING_RATE = config.LEARNING_RATE

# Other const variables
N_WORKERS = config.N_WORKERS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARCHITECTURE = config.ARCHITECTURE
IMG_SIZE = config.IMG_SIZE
PIN_MEMORY = config.PIN_MEMORY

PRETRAINED = True

#%% Dataloader

class ClassifierDataset(torch.utils.data.Dataset):
    def __init__(
        self, transform=None, data_path="/dtu/datasets1/02514/isic"
    ):
        self.mask_paths = []
        self.back_image_paths = sorted(glob.glob(f"{data_path}/background/*.jpg"))
        self.fore_image_paths = sorted(glob.glob(f"{data_path}/train_allstyles/Images/*.jpg"))

        test1 = {path:0 for path in self.back_image_paths}
        test2 = {path:1 for path in self.fore_image_paths}

        self.labels = list(test1.values())+list(test2.values())
        self.image_paths = list(test1.keys())+list(test2.keys())

        self.transform = transform


    def __len__(self):
        "Returns the total number of samples"
        return len(self.image_paths)

    def __getitem__(self, idx):
        "Generates one sample of data"
        target = self.labels[idx]

       
        image_path = self.image_paths[idx]
        

        image = np.array(Image.open(image_path))
        

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image, target



size = 224

train_transform = A.Compose(
    [
        A.Resize(*IMG_SIZE),
        # A.Rotate(limit=35, p=1.0),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

test_transform = A.Compose(
    [
        A.Resize(*IMG_SIZE),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

trainset = ClassifierDataset(transform=train_transform)
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
testset = ClassifierDataset(transform=test_transform)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=3)


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
                nn.Linear(224*224*16, 500),
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


       
#%% train

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

num_epochs = EPOCHS

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
                                                        epoch=epoch, total=EPOCHS,
                                                        test=100*test_acc, train=100*train_acc,
                                                        loss_train=loss_train, loss_test=loss_test))
    
