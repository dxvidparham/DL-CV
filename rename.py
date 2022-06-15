#%%
import os

start_path = os.getcwd()

beard_path = start_path + "/no_beard_copy"

print(beard_path)
# %%
idx = 0
for root, subdirectories, files in os.walk(beard_path):
    for subdirectory in subdirectories:
        pass
        # print(os.path.join(root, subdirectory))
    for file in files:
        if "npz" in file:
            old_name = os.path.join(root, file)
            print(os.path.join(root, file))
            new_name = 'no_beard_' + str(idx)+'.npz'
            new_name = os.path.join(root, new_name)
            os.rename(old_name,new_name)
            idx+=1
            # print(os.path.join(root, file))

# %%

for root, subdirectories, files in os.walk(beard_path):
    for subdirectory in subdirectories:
        pass
        # print(os.path.join(root, subdirectory))
    for file in files:
        if "npz" in file:
            
            print(os.path.join(root, file))

# %%
