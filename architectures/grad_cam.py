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

        print(self.vgg)
        
        # disect the network to access its last convolutional layer
        self.features_conv = self.vgg.features[:-1]
        
        # get the max pool of the features stem
        self.max_pool = self.vgg.features[-1]
        self.avgpool = self.vgg.avgpool
        
        # get the classifier of the vgg19
        self.classifier = self.vgg.classifier
        self.classifier[-1] = nn.Linear(in_features=self.vgg.classifier[-1].in_features,
                                        out_features=2
                                        )
        
        # placeholder for the gradients
        self.gradients = None

        self.tensorhook = []
        self.layerhook = []

        self.selected_out = None

        self.layerhook.append(self.features_conv.register_forward_hook(self.forward_hook()))


        for p in self.vgg.parameters():
            p.requires_grad = True

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))
        return hook
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        self.feature_out = x

        # apply the remaining pooling
        x = self.max_pool(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, self.selected_out
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)


#----------------------------------------------------------------------------
# class VGG_seg(nn.Module):
#     def __init__(self):
#         super(VGG_seg, self).__init__()
        
#         # disect the network to access its last convolutional layer
#         self.features_conv = nn.Sequential(
#                 nn.Conv2d(in_channels=3,
#                         out_channels=8,
#                         kernel_size=(3,3),
#                         padding='same'),
#                 nn.ReLU(),
#                 nn.Conv2d(in_channels=8,
#                           out_channels=8,
#                           kernel_size=(3,3),
#                          padding='same'),
#                 nn.ReLU(),
#                 nn.Conv2d(in_channels=8,
#                           out_channels=16,
#                           kernel_size=(3,3),
#                          padding='same'),
#         )
        
        
#         # get the classifier of the vgg19
#         self.classifier = nn.Sequential(
#                 nn.Linear(224*224*16, 500),
#                 nn.ReLU(),
#                 nn.Linear(500, 2),
#         )
        

#         # placeholder for the gradients
#         self.gradients = None

#         self.tensorhook = []
#         self.layerhook = []

#         self.selected_out = None

#         self.layerhook.append(self.features_conv.register_forward_hook(self.forward_hook()))


#         for p in self.features_conv.parameters():
#             p.requires_grad = True

#         for p in self.classifier.parameters():
#             p.requires_grad = True

#     # hook for the gradients of the activations
#     def activations_hook(self, grad):
#         self.gradients = grad

#     def get_act_grads(self):
#         return self.gradients

#     def forward_hook(self):
#         def hook(module, inp, out):
#             self.selected_out = out
#             self.tensorhook.append(out.register_hook(self.activations_hook))
#         return hook
        
#     def forward(self, x):
#         x = self.features_conv(x)
        
#         # register the hook
#         self.feature_out = x
#         # h = x.register_hook(self.activations_hook)

#         # apply the remaining pooling
#         # x = self.max_pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x


    
#     # method for the gradient extraction
#     def get_activations_gradient(self):
#         return self.gradients
    
#     # method for the activation exctraction
#     def get_activations(self, x):
#         return self.features_conv(x)

#     def segmentation(self, x):
#         inx = x
#         x = self.forward(x)
#         pred = F.sigmoid(x).argmax(dim = 1)

#------------------------------------------------------------------------------

# class VGG_seg(nn.Module):
#     def __init__(self):
#         super(VGG_seg, self).__init__()
        
#         self.convolutional = nn.Sequential(
#                 nn.Conv2d(in_channels=3,
#                         out_channels=8,
#                         kernel_size=(3,3),
#                         padding='same'),
#                 nn.ReLU(),
#                 nn.Conv2d(in_channels=8,
#                           out_channels=8,
#                           kernel_size=(3,3),
#                          padding='same'),
#                 nn.ReLU(),
#                 nn.Conv2d(in_channels=8,
#                           out_channels=16,
#                           kernel_size=(3,3),
#                          padding='same'),
#         )

#         self.fully_connected = nn.Sequential(
#                 nn.Linear(224*224*16, 500),
#                 nn.ReLU(),
#                 nn.Linear(500, 2),
#         )
    
#     def forward(self, x):
#         x = self.convolutional(x)
#         #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
#         x = x.view(x.size(0), -1)
#         x = self.fully_connected(x)
#         return x
    

#%%

if __name__ == "__main__":
    print("in main")
    import sys
    sys.path.append("../")
    from dataLoader import ISICDataset

    model = VGG_seg()
    print(model)
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
            batch_size=16,
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
