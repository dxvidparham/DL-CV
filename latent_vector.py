#%%
import numpy as np
from sklearn import svm

import os

import matplotlib.pyplot as plt



# load npz
workdir = os.getcwd()
beard_path = workdir + "/beard_nobeard"

c0 = [] # beard
c1 = [] # no beard

for root, subdirectories, _ in os.walk(beard_path):
    for subdirectory_name in subdirectories:
        # print(os.path.join(root, subdirectory_name))
        subdir_path = os.path.join(root, subdirectory_name)
        # print(subdirectory_name)
        for r, _, files in os.walk(subdir_path):
            if subdirectory_name == "beard":
                for f in files:
                    file_path = os.path.join(r,f)
                    w = np.load(file_path)['w']
                    c0.append(w)
                
            elif subdirectory_name == "no_beard":
                for f in files:
                    file_path = os.path.join(r,f)
                    w = np.load(file_path)['w']
                    c1.append(w)
            else:
                print('something went wrong')

c0 = np.array(c0)
c1 = np.array(c1)

c0_reshape = c0.reshape((c0.shape[0],c0.shape[1]*c0.shape[2]*c0.shape[3]))
c1_reshape = c1.reshape((c1.shape[0],c1.shape[1]*c1.shape[2]*c1.shape[3]))

x = np.concatenate((c0_reshape,c1_reshape), axis=0)

target = np.append(np.zeros(len(c0)),np.ones(len(c1)))

#%%

# c0 = np.random.random((15,18,512))
# c0[:,1] = c0[:,1] + 3 

# c0_reshape = c0.reshape((c0.shape[0],c0.shape[1]*c0.shape[2]))

# c1 = np.random.random((15,18,512))

# c1_reshape = c1.reshape((c1.shape[0],c1.shape[1]*c1.shape[2]))

#%%

clf = svm.SVC(kernel='linear')
clf.fit(x, target)


# get support vectors
print(clf.support_vectors_)


# get indices of support vectors
print(clf.support_)

# get number of support vectors for each class
print(clf.n_support_)

#%%

plt.scatter(x[:, 0], x[:, 1], c=target, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()

# plot support vectors
ax.scatter(
    clf.support_vectors_[1, 0],
    clf.support_vectors_[1, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)
plt.show()
# %%

print('coef')
print(clf.coef_)

print(clf.coef_.shape)

w_beard = np.array(clf.coef_).reshape((1,c1.shape[2],c1.shape[3]))

print(w_beard.shape)

np.save('targets/w_beard',w_beard)

# %%


