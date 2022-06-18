#%%

import torch
import torch.nn as nn
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2


from dataLoader import ISICDataset

IMG_SIZE = [64,64]

#%% Network

model = torchvision.models.segmentation.fcn_resnet101(pretrained=True)


#%%

# model.aux_logits = False

# model.fc = nn.Sequential(
#     nn.Linear(model.fc.in_features,10),
#     nn.Linear(10,2)
# )

#%%

if __name__ == "__main___":

#%%

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
    training_set = ISICDataset(train_transform)
    trainloader = torch.utils.data.DataLoader(
            training_set,
            shuffle=True,
        )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Push model to GPU if available
    if torch.cuda.is_available():
        print(f"[INFO] Training model on GPU ({torch.cuda.get_device_name(0)})...")
        model.to(DEVICE)
    else:
        print("[INFO] Training model on CPU...")

    for images, labels in trainloader:
        images = images.float().to(DEVICE)
        # labels = labels.float().unsqueeze(1).to(DEVICE)


        output = model(images)

        print(output)
# %%
