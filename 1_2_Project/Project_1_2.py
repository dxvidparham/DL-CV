#%%
import os
import numpy as np
import PIL.Image as Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from pycocotools.coco import COCO

import time

from omegaconf import OmegaConf

#%% Check if torch available
if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%% load config
DIR= os.path.dirname(os.path.realpath(__file__))

config = OmegaConf.load(f"{DIR}/config/project_1_2.yaml")

_num_epoch = config.epoch
_batch_size = config.batch_size
_learning_rate = config.learning_rate

SIZE = config.size

N_WORKERS = 1

#%% wandb init
use_wandb =  False

if use_wandb == True:
    import wandb
    wandb.init(project="Project1", entity="dlincvg1", config=dict(config),)



#%% Dataloader
from torchvision.io import read_image

class TacoDataset(torch.utils.data.Dataset):
    def __init__(self, augment=False, size=1000):
        self.root = "/dtu/datasets1/02514/data_wastedetection/"
        self.annotation = f"{self.root}/annotations.json"
        self.coco = COCO(self.annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.augment = augment
        if self.augment:
            self.transform = transforms.Compose(
                [
                    # transforms.Resize((size, size)),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    # transforms.Resize((size, size)),
                    transforms.ToTensor(),
                ]
            )

    def crop_img(self, img, ymin, ymax, xmin, xmax):
        return img[:, ymin:ymax, xmin:xmax]

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # print(coco_annotation)
        
        # path for input image
        path = coco.loadImgs(img_id)[0]["file_name"]
        # open the input image
        img = Image.open(os.path.join(self.root, path))
        # print(img)

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]["bbox"][0]
            ymin = coco_annotation[i]["bbox"][1]
            xmax = xmin + coco_annotation[i]["bbox"][2]
            ymax = ymin + coco_annotation[i]["bbox"][3]
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)

        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = [coco_annotation[i]["area"] for i in range(num_objs)]
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {
            "boxes": boxes,
            "labels": labels,
            "image_id": img_id,
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transform is not None:
            img = self.transform(img)


        # print('img:',img.shape)
        # print('box:',my_annotation["boxes"].shape)
        # print('labels:',my_annotation["labels"].shape)
        # print('image_id:',my_annotation["image_id"].shape)
        # print('area:',my_annotation["area"].shape)
        # print('iscrowd:',my_annotation["iscrowd"].shape)
        # print("")


        return img, my_annotation

    def __len__(self):
        return len(self.ids)
    
    
#%% 

# a simple custom collate function, just to show the idea
def my_collate(batch):
    data = [item[0] for item in batch]
    my_annotation = [item[1] for item in batch]
    # target = torch.LongTensor(target)

    return [data, my_annotation]

print("[INFO] Load datasets from disk...")
training_set = TacoDataset()

testing_set = TacoDataset()

print("[INFO] Prepare labeldataloaders...")
trainloader = torch.utils.data.DataLoader(
    training_set, shuffle=False, batch_size=_batch_size, collate_fn = my_collate
)
testloader = torch.utils.data.DataLoader(
    testing_set, shuffle=False, num_workers=N_WORKERS, batch_size=_batch_size
)


# select device (whether GPU or CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# # DataLoader is iterable over Dataset
# for imgs, annotations in trainloader:
#     imgs = list(img.to(device) for img in imgs)
#     annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
#     print(annotations)


trainloader_iter = iter(trainloader)
x, y = next(trainloader_iter)


#%% Network

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.convolutional = nn.Sequential(
                nn.Conv2d(in_channels=3,
                        out_channels=8,
                        kernel_size=(3,3),
                        padding='same'),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.Conv2d(in_channels=8,
                          out_channels=8,
                          kernel_size=(3,3),
                         padding='same'),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.Conv2d(in_channels=8,
                          out_channels=16,
                          kernel_size=(3,3),
                         padding='same'),
                nn.BatchNorm2d(16),
        )

        self.fully_connected = nn.Sequential(
                nn.Linear(128*128*16, 500),
                nn.ReLU(),
                nn.Linear(500, 27),
        )
    
    def forward(self, x):
        x = self.convolutional(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x


model = Network()
model.to(device)
if use_wandb == True:
    wandb.watch(model, log_freq=100)


#%% train

optimizer = torch.optim.Adam(model.parameters(), lr=_learning_rate)
criterion = nn.CrossEntropyLoss()

num_epochs = _num_epoch

start_t = time.time()
best_val = 100000000

for epoch in tqdm(range(num_epochs), unit='epoch'):
    #For each epoch
    train_correct = 0
    model.train()
    for minibatch_no, (imgs, annotations) in tqdm(enumerate(trainloader), total=len(trainloader)):
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        
        

        # get the proposal frames for the image_id

        # crop and resize
        # transforms.functional.crop(img, top, left, height, width)
        transforms.Resize(SIZE)
        

        # ----------------
        # Balancing the input
        # ----------------
        # Get groud truth
        # What if there is no classes?
        # -> Train on background.

        # compute the IoU

        # -> crops of the classes & crops of the backgroud

        # balacing by percentage

        # create batches of the 1 image proposals
        # create the tagets

        # predict the output
        predicted_labels = model(balanced_data_batch)


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
    
    # =====================================================
    # Test
    # =====================================================
    with torch.no_grad():
        model.eval()
        for imgs, annotations in testloader:

            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            
            # get proposals
            
            # crop & resize


            for prosals_batch in cropped_proposals:
            
                batch_output = model(proposals_batch)

                output.append(batch_output)

            singel_classes_boxes = NMS(output)
            test_score = IoU(single_classes_boxes, GroundTruth)


            loss_test = criterion(output, target)

            predicted = output.argmax(1)
            
            test_correct += (target==predicted).sum().item()
        train_acc = train_correct/len(trainset)
        test_acc = test_correct/len(testset)
        
        
        
    print("\n \n Epoch: {epoch}/{total} \n Accuracy train: {train:.1f}%\t test: {test:.1f}% \n Loss: \t train: {loss_train:.3f} \t test: {loss_test:.3f}\n".format(
                                                        epoch=epoch, total=_num_epoch,
                                                        test=100*test_acc, train=100*train_acc,                                               loss_train=loss_train, loss_test=loss_test))
    
    if use_wandb==True:
        wandb.log({"train_acc":train_acc})
        wandb.log({"test_acc":test_acc})

        wandb.log({"train_loss":loss_train})
        wandb.log({"test_loss":loss_test})
        
        
    # Save best model if val_loss in current epoch is lower than the best validation loss
    if loss_test < best_val:
        best_val = loss_test
        print("\n[INFO] Saving new best_model...\n")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            'models/',
        )
        
    end_t = time.time()
    run_time = end_t - start_t

    print(
        f"[INFO] Successfully completed training session. Running time: {run_time/60:.2f} min"
    )
