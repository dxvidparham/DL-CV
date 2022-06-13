#%%
import os
from matplotlib import image
import numpy as np
import PIL.Image as Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.sampler import WeightedRandomSampler

import time

from omegaconf import OmegaConf


from dataLoader import TacoDataset
from utils import collate_wrapper
from models.baseline import Network

#%% Check if torch available
if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%% load config
DIR= os.path.dirname(os.path.realpath(__file__))

config = OmegaConf.load(f"{DIR}/config/config.yaml")

EPOCHS = config.EPOCHS
LEARNING_RATE = config.LEARNING_RATE

# Optimizer Hyperparameter
BATCH_SIZE = config.BATCH_SIZE
print("Batch_size:", BATCH_SIZE)

# config  variables
N_WORKERS = config.N_WORKERS

SPLIT_DISTRIBUTION = config.SPLIT_DISTRIBUTION
_strategy = 0

SIZE = config.IMG_SIZE

# N_WORKERS = 1

#%% wandb init
use_wandb =  False

if use_wandb == True:
    import wandb
    wandb.init(project="Project1", entity="dlincvg1", config=dict(config),)

#=============================================
#%% Load Taco Data
#=============================================

dataset = TacoDataset()
dataset_size = dataset.__len__()
train_count = int(SPLIT_DISTRIBUTION.train[_strategy] * dataset_size)
test_count = int(SPLIT_DISTRIBUTION.test[_strategy] * dataset_size)
val_count = int(train_count * config.VALIDATION_SIZE)
train_count -= val_count

train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_count, test_count, val_count]
)

trainloader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    num_workers=N_WORKERS,
    batch_size=BATCH_SIZE,
    collate_fn=collate_wrapper,
)

testloader = torch.utils.data.DataLoader(
    test_dataset,
    num_workers=N_WORKERS,
    batch_size=BATCH_SIZE,
    collate_fn=collate_wrapper,
)

valloader = torch.utils.data.DataLoader(
    val_dataset,
    num_workers=N_WORKERS,
    batch_size=BATCH_SIZE,
    collate_fn=collate_wrapper,
)

#============================================
#%% Loader traing Proposal dataset
#============================================

# for data, annotations in trainloader:
#     for i in range(len(data)):
#         img = data[i]
#         my_anno = annotations[i]
#         propset = ProposalDataset(img,my_anno,SIZE)


#     propset = ProposalDataset(img,my_annotation,SIZE)

#     weight = 1. / propset.class_sample_count

#     target = propset.target
#     # target = torch.from_numpy(target).long()

#     samples_weight = np.array([weight[t] for t in target])

#     samples_weight = torch.from_numpy(samples_weight)
#     samples_weigth = samples_weight.double()
#     sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

#     proploader = DataLoader(
#         propset, shuffle=False, batch_size=_batch_size, sampler=sampler,
#     )

#     proploader_total += proploader


            
# ===============================================
#%% Load Network
# ===============================================

model = Network()
model.to(device)
if use_wandb == True:
    wandb.watch(model, log_freq=100)

#%%
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def get_proposals(image_path='000000.txt'):
    for jpg in ["jpg","JPG"]:
            if jpg in image_path:
                prop_name = image_path.replace(jpg,"txt")
    annotation_path = f"{DIR}/project2_ds_new/"+prop_name
    
    prop_anno_list = []
    with open(annotation_path) as f:
        proposal_list = f.readlines()
        for box_str in proposal_list:
            prop_box = box_str.split()
            prop_box = [int(i) for i in prop_box]
            prop_anno_list.append(prop_box)
    return prop_anno_list


def crop_and_resize(img, proposals, size):
        # crop & resize
        img_list = []
        for prop in proposals:
            prop_crop = prop_crop = transforms.functional.crop(img, prop[0],prop[1],prop[3]-prop[1],prop[2]-prop[0])
            img_list.append(resized = transforms.functional.resize(prop_crop,(size, size)))
        return img_list


#%% ===================================================================
# train
# =====================================================================

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

start_t = time.time()
best_val = 100000000

def unique(list1):
    x = list1.cpu().numpy()
    return np.unique(x)
      

for epoch in tqdm(range(EPOCHS), unit='epoch'):
    #For each epoch
    train_correct = 0
    model.train()
    for minibatch_no, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        # imgs = list(img.to(device) for img in imgs)
        
        for i in range(len(data)):
            img = data[0][i]
            my_anno = data[1][i]
            propset = ProposalDataset(img,my_anno,SIZE)

            weight = 1. / propset.class_sample_count

            target = propset.target
            target = [0 if x==0 else 1 for x in target]
            # target = torch.from_numpy(target).long()

            samples_weight = np.array([weight[t] for t in target])

            samples_weight = torch.from_numpy(samples_weight)
            samples_weigth = samples_weight.double()
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

            proploader = DataLoader(
                propset, shuffle=False, batch_size=BATCH_SIZE, sampler=sampler,
            )

            for prop_batch_no, (prop_img,prop_anno) in tqdm(enumerate(proploader), total=len(proploader)):
                # print("prop loader worked")
                prop_img = prop_img.to(device)
                # print(prop_anno.items())
                # prop_anno =  [{k: v.to(device) for k, v in t.items()} for t in prop_anno]
                #Zero the gradients computed for each weight
                optimizer.zero_grad()
                #Forward pass your image through the network
                predicted = model(prop_img)
                #Compute the loss
                labels = (prop_anno["labels"]).to(device)
                loss_train = criterion(predicted, labels)
                #Backward pass through the network
                loss_train.backward()
                #Update the weights
                optimizer.step()
                
                #Compute how many were correctly classified
                predicted_class = predicted.argmax(1)
                train_correct += (labels==predicted_class).sum().cpu().item()

            break
        break
            # print(train_correct)

    # =====================================================
    # Test
    # =====================================================
    with torch.no_grad():
        model.eval()
        map_input_list = []
        gt_list = []
        for data in testloader:

            # imgs = list(img.to(device) for img in imgs)
            # annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            test_correct = []
            for i in range(len(data)):
                img = data[0][i]
                my_anno = data[1][i]
                
                testpropset = ProposalDataset(img,my_anno,SIZE)
                testproploader = DataLoader(
                    testpropset, shuffle=False, batch_size=BATCH_SIZE
                )
                
                output = []
                targets = []
                loss_test = []
                prop_boxes = []
                
                
                for prop_batch_no, (prop_img,prop_anno) in tqdm(enumerate(testproploader), total=len(testproploader)):
                    labels = (prop_anno["labels"]).to(device)
                    prop_box = prop_anno["boxes"]
                    prop_boxes.append(prop_box)
                    img_id = prop_anno["image_id"]
                    prop_img = prop_img.to(device)
                    batch_output = model(prop_img)
                    # print(batch_output.shape)
                    output.append(batch_output)
                    targets.append(labels)

                    batch_loss_test = criterion(batch_output, labels)
                    loss_test.append(batch_loss_test)
                gt_input = [[img_id, lab, 1, bb[0],bb[1],bb[2],bb[3]] for lab,bb in zip(my_anno["labels"],my_anno["boxes"].cpu().numpy())]
                gt_list.append(gt_input)
                output_stack = torch.cat(output, dim=0)
                print(output_stack.shape)
                predicted = output_stack.argmax(1)

                unique_predicted = unique(predicted)
                bbox_uniq_list = []
                NMS_list = []

                for uniq in unique_predicted:
                    for idx in range(len(prop_boxes)):
                        if uniq == predicted[idx]:
                            # prop_boxes=prop_boxes.cpu().numpy()
                            print(prop_boxes[idx])
                            bbox_uniq = [uniq, output[idx],
                            prop_boxes[idx][0],prop_boxes[idx][1], prop_boxes[idx][2],
                            prop_boxes[idx][3]]
                            bbox_uniq_list.append(bbox_uniq)
                print(bbox_uniq_list[0])
                NMS_list.append(NMS(bounding_boxes= bbox_uniq_list, 
                                    iou_threshold=0.5, threshold=0.2))
            train_acc = train_correct/len(data)
            test_acc = test_correct/len(data)
            map_input = [[img_id, ii[0],ii[1],ii[2],ii[3],ii[4],ii[5]] for ii in NMS_list]
            map_input_list.append(map_input)
            
            test_correct += (target==predicted).sum().item()

        mAP = mAP(map_input_list, gt_input)
        print(mAP)
        

        
    
    
    
    # # =====================================================
    # # Test
    # # =====================================================

    # # Comput the test accuracy
    # test_correct = 0

    # with torch.no_grad():
    #     model.eval()
    #     for imgs, annotations in testloader:

    #         imgs = list(img.to(device) for img in imgs)
    #         annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            
    #         # get proposals
            
    #         # crop & resize


    #         for prosals_batch in cropped_proposals:
            
    #             batch_output = model(proposals_batch)

    #             output.append(batch_output)

    #         singel_classes_boxes = NMS(output)
    #         test_score = IoU(single_classes_boxes, GroundTruth)


    #         loss_test = criterion(output, target)

    #         predicted = output.argmax(1)
            
    #         test_correct += (target==predicted).sum().item()
    #     train_acc = train_correct/len(trainset)
    #     test_acc = test_correct/len(testset)
        
        
        
    print("\n \n Epoch: {epoch}/{total} \n Accuracy train: {train:.1f}%\t test: {test:.1f}% \n Loss: \t train: {loss_train:.3f} \t test: {loss_test:.3f}\n".format(
                                                        epoch=epoch, total=EPOCHS,
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

# %%
