#%%
import torch
import torch.nn as nn
from torch.utils import data
import torchvision
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.nn.functional as F


#%%

class VGG_seg(nn.Module):
    def __init__(self):
        super(VGG_seg, self).__init__()

        # get the pretrained VGG19 network
        self.vgg = torchvision.models.vgg19(pretrained=True)
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:36]
        
        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        # get the classifier of the vgg19
        self.classifier = self.vgg.classifier
        self.classifier[-1] = nn.Linear(in_features=self.vgg.classifier[-1].in_features,
                                        out_features=2
                                        )
        
        # placeholder for the gradients
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        self.feature_out = x
        self.h = x.register_hook(self.activations_hook)

        
        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)

    def segmentation(self, x):
        inx = x
        x = self.forward(x)
        pred = F.sigmoid(x).argmax(dim = 1)


        


#%%

if __name__ == "__main__":
    print("in main")
    import sys
    sys.path.append("../")
    from dataLoader import ISICDataset

    model = VGG_seg()
    IMG_SIZE = [240, 160] 


    train_transform = A.Compose(
            [
                A.Resize(*[224, 224]),
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
            num_workers=4,
            batch_size=2,
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

        # Set the requires_grad_ to the image for retrieving gradients
        images.requires_grad_()

        output = model(images)
        # Catch the output
        output_idx = output.argmax()
        output_max = output[0, output_idx]

        # Do backpropagation to get the derivative of the output based on the image
        output_max.backward()

        # Retireve the saliency map and also pick the maximum value from channels on each pixel.
        # In this case, we look at dim=1. Recall the shape (batch_size, channel, width, height)
        saliency, _ = torch.max(images.grad.data.abs(), dim=1) 
        saliency = saliency.reshape(224, 224)

        # Reshape the image
        image = images.reshape(-1, 224, 224)

        # Visualize the image and the saliency map
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
        ax[0].axis('off')
        ax[1].imshow(saliency.cpu(), cmap='hot')
        ax[1].axis('off')
        plt.tight_layout()
        fig.suptitle('The Image and Its Saliency Map')
        plt.show()
        plt.savefig('img.png')


        seg_pred = model.segmentation(images)

        print(output.shape)
# %%
