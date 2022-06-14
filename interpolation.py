from dataclasses import dataclass
import torch
import pickle
import PIL.Image
import numpy as np


outdir = "out"
img_name = "test_alex"

V1 = "./out/alex.npz"
V2 = "./out/chris.npz"

# # https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
with open('model/ffhq.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
z = torch.randn([1, G.z_dim]).cuda()    # latent pytcodes
c = None                                # class labels (not used in this example)
img = G(z, c)                           # NCHW, float32, dynamic range [-1, +1]


v1 = np.load(V1)['w']
v2 = np.load(V2)['w']



# print(torch.tensor(v1))
v1 = torch.tensor(v1)
v2 = torch.tensor(v2)

img = G(v1, c)
img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
img_1 = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/{img_name}.png')