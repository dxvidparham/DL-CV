import os
import subprocess
# subprocess.Popen("script2.py 1", shell=True)

<<<<<<< HEAD
images_dir = "face_img/beard_aligned/beards"

for img in os.listdir(images_dir):

    print(f"******************** procesing file: {img} **************************")
    os.system(f"python projector.py --outdir=out/beard --target=face_img/beard_aligned/beards/{img}  --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl --num-steps=500")
=======
images_dir = "./targets/beard_aligned/no_beard"

for img in os.listdir(images_dir):

    print(f"******************** procesing file: beard_aligned/{img} **************************")
    os.system(f"python projector.py --outdir=out/no_beard/{img} --target=targets/beard_aligned/no_beard/{img} --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl --save-video=False --num-steps=500")
>>>>>>> 2b04f560c518a593bb3f6e64935660945ad2185c
    # subprocess.Popen(f"python projector.py --outdir=out --target=face_img/beard_aligned/{img}  --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl", shell=False)
    