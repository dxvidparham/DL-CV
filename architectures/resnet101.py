#%%

import torch
import torch.nn as nn
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2


# from dataLoader import ISICDataset

#%% Network

# def load_resnet():
#     resnet101 = torchvision.models.segmentation.fcn_resnet101(pretrained=True)

#     resnet101.classifier[4] = nn.Conv2d(in_channels=resnet101.classifier[4].in_channels,
#                                         out_channels= 1,
#                                         kernel_size=resnet101.classifier[4].kernel_size,
#                                         stride=resnet101.classifier[4].stride)

#     return resnet101


class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self): 
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = torchvision.models.segmentation.fcn_resnet101(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(in_channels=self.model.classifier[4].in_channels,
                                        out_channels= 1,
                                        kernel_size=self.model.classifier[4].kernel_size,
                                        stride=self.model.classifier[4].stride)
        
    def forward(self, x):
        return self.model(x)["out"]


#%%

# model.aux_logits = False

# model.fc = nn.Sequential(
#     nn.Linear(model.fc.in_features,10),
#     nn.Linear(10,2)
# )

#%%
if __name__ == "__main__":

    import sys

    sys.path.append("../")

    from dataLoader import ISICDataset

    model = SegmentationModelOutputWrapper()
    IMG_SIZE = [240, 160] 


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

        print(output.shape)
# %%
