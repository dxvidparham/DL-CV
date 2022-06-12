#%%
import os
import numpy as np 
import skimage 
import skimage.io as skio
from skimage.transform import resize
from skimage.io import imread_collection
import selective_search
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# %%
ds_dir = "project2_ds/"
for folder_name in os.listdir(ds_dir):
    batch_dir = ds_dir+folder_name+"/"
    if os.path.isdir(batch_dir):
        print(f"Processing {batch_dir} ******************")
        for file_name in os.listdir(batch_dir):
            # print(file_name[-4:])
            if file_name[-4:] != ".txt":
                id_name = file_name[:-4]
                image = skio.imread(batch_dir + file_name)
                max_dim = max(image.shape)
                scale = 500/max_dim
                image = resize(image,[int(image.shape[0]*scale),int(image.shape[1]*scale)])
                boxes = selective_search.selective_search(image, mode='fast')
                # print("got boxes")
                boxes_filter = selective_search.box_filter(boxes, min_size=20, topN=1000)
                boxes_filter_arr_scaled = np.array(boxes_filter)/scale
                np.savetxt(batch_dir+id_name+".txt", boxes_filter_arr_scaled, fmt="%d")
                print("saved txt", batch_dir+id_name+".txt")

# print(len(image_coll),image_coll[0])


#%%



# drawing rectangles on the original image
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(image)
for x1, y1, x2, y2 in boxes_filter:
    bbox = mpatches.Rectangle(
        (x1, y1), (x2-x1), (y2-y1), fill=False, edgecolor='green', linewidth=1)
    ax.add_patch(bbox)
plt.axis('off')
plt.show()
print(boxes_filter)
input()
print(len(boxes), len(boxes_filter))