import os
import subprocess
# subprocess.Popen("script2.py 1", shell=True)

images_dir = "face_img/beard_aligned/"

for img in os.listdir(images_dir):

    print(f"******************** procesing file: beard_aligned/{img} **************************")
    os.system(f"python projector.py --outdir=out/beard --target=face_img/beard_aligned/{img}  --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl")
    # subprocess.Popen(f"python projector.py --outdir=out --target=face_img/beard_aligned/{img}  --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl", shell=False)
    break