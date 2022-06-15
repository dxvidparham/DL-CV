import os
import subprocess
# subprocess.Popen("script2.py 1", shell=True)

images_dir = "face_img/beard_aligned/beards"

for img in os.listdir(images_dir):

    print(f"******************** procesing file: {img} **************************")
    os.system(f"python projector.py --outdir=out/beard --target=face_img/beard_aligned/beards/{img}  --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl --num-steps=500")
    # subprocess.Popen(f"python projector.py --outdir=out --target=face_img/beard_aligned/{img}  --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl", shell=False)
    