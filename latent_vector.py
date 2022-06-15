#%%
import numpy as np

from sklearn import svm
# from sklearn.inspection import DecisionBoundaryDisplay

import matplotlib.pyplot as plt


c0 = np.random.random((15,18,512))
c0[:,1] = c0[:,1] + 3 
c1 = np.random.random((15,18,512))

x = np.concatenate((c0,c1), axis=0)

target = np.append(np.zeros(15),np.ones(15))

clf = svm.SVC()
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

print('dual_coef', clf.dual_coef_)


# %%
