#%%
from dataclasses import dataclass
from matplotlib.transforms import Bbox
import torch
import pickle
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt 


outdir = "out"
img_name = "old_david"

V1 = "./out/latents/projected_w_david.npz"
V2 = "./stylegan2directions/gender.npy"   # NCHW, float32, dynamic range [-1, +1]
# V2 = "./targets/w_beard.npy"

input_type = "david"
direction_type = "gender"

v1 = np.load(V1)['w']
direction = np.load(V2)
direction = direction.reshape((v1.shape[1],v1.shape[2]))

print(v1.shape)
print(direction.shape)
# input()
#%%
# # https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
with open('model/ffhq.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
z = torch.randn([1, G.z_dim]).cuda()    # latent pytcodes
c = None                                # class labels (not used in this example)
# img = G(z, c)    
#%%
assert v1.shape[1:] == (G.num_ws, G.w_dim)

# print(torch.tensor(v1))
v1 = torch.tensor(v1).cuda()
direction = torch.tensor(direction).cuda()
for idx, w in enumerate(v1):
    img = G.synthesis(w.unsqueeze(0), noise_mode="const")
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    x_alex = img[0].cpu().numpy()
    # img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/Alex_projection{idx:02d}.png')


plt.figure(figsize=(20,20))
subplots = [plt.subplot(2, 5, k+1) for k in range(5)]
subplots[0].imshow(x_alex)
subplots[0].set_title('Original')
subplots[0].axis('off')

coef = 3

for k in range(1,5):
    new_latent_vector = v1.clone()
    new_latent_vector = (v1 + (k*coef)*direction)
    x_k = G.synthesis(new_latent_vector[0].unsqueeze(0), noise_mode="const")
    x_k = (x_k.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    subplots[k].imshow(x_k[0].cpu().numpy())
    subplots[k].set_title(str(direction_type) +" coef=" +str(coef*k))
    subplots[k].axis('off')
plt.savefig(f"out/{direction_type}_{input_type}_{coef}.png",bbox_inches = "tight")

# def move_and_show(latent_vector, direction, coeffs):
#     fig,ax = plt.subplots(1, len(coeffs), figsize=(15, 10), dpi=80)
#     for i, coeff in enumerate(coeffs):
#         new_latent_vector = latent_vector.copy()
#         new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
#         ax[i].imshow(generate_image(new_latent_vector))
#         ax[i].set_title('Coeff: %0.1f' % coeff)
#     [x.axis('off') for x in ax]
#     plt.show()
# %%
