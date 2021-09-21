import numpy as np
import torch
import copy
from sklearn.manifold import TSNE
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib as mpl
#im_mat1 = loadmat('G:\hyperspectral image dataset\public dataset\pavia\PaviaU.mat') #Switch to own directory
#image1 = im_mat1['paviaU']
#print(im_mat1.keys())
im_mat1 = loadmat('G:\hyperspectral image dataset\public dataset\IndianPines\Indian_pines_corrected.mat') #Change to own directory
image1 = im_mat1['indian_pines_corrected']
#im_mat2 = loadmat('G:\hyperspectral image dataset\public dataset\pavia\PaviaU_gt.mat')
#image2 = im_mat2['paviaU_gt']
im_mat2 = loadmat('G:\hyperspectral image dataset\public dataset\IndianPines\Indian_pines_gt.mat')
image2 = im_mat2['indian_pines_gt']
print(im_mat2.keys())
d = np.zeros_like(image2)
d[image2 > 0] = True
d = torch.from_numpy(d)
d = d.numpy()
mask1 = copy.deepcopy(d)
mask1 = mask1.astype(np.bool)
mask = copy.deepcopy(d)
mask = mask.astype(np.bool)
image1=image1.transpose(2,0,1)
print(image1.shape)
for i in range(200): #Change to the bands of the dataset e.g. 200,103,144
    img = image1[i, :, :]
    img = img.astype(np.float32)
    img = torch.from_numpy(img).squeeze(0)
    # print(mask.shape)
    img = img.numpy()
    b = img[mask]
    b = b.reshape(1, -1)
    # print(b.shape)
    if i == 0:
        out = b
        continue
    out = np.concatenate((out, b), 0)
    print(out.shape)
out = out.transpose(1, 0)

y = image2[mask1]
print(y.shape)


def t_sne(latent_vecs, y):

    latent_vecs_reduced = TSNE(n_components=2, random_state=0).fit_transform(latent_vecs)
    # latent_vecs_reduced = PCA(n_components=2).fit_transform(latent_vecs)
    print(latent_vecs_reduced.shape)
    for j in range(latent_vecs_reduced.shape[0]):
        if y[j] == 1:
            plt.scatter(latent_vecs_reduced[j, 0], latent_vecs_reduced[j, 1], marker='o', c='red')
        elif y[j] == 2:
            plt.scatter(latent_vecs_reduced[j, 0], latent_vecs_reduced[j, 1],marker='o', c='green')
        elif y[j] == 3:
            plt.scatter(latent_vecs_reduced[j, 0], latent_vecs_reduced[j, 1],marker='o', c='yellow')
        elif y[j] == 4:
            plt.scatter(latent_vecs_reduced[j, 0], latent_vecs_reduced[j, 1], marker='o', c='maroon')
        elif y[j] == 5:
            plt.scatter(latent_vecs_reduced[j, 0], latent_vecs_reduced[j, 1], marker='o', c='black')
        elif y[j] == 6:
            plt.scatter(latent_vecs_reduced[j, 0], latent_vecs_reduced[j, 1], marker='o', c='cyan')
        elif y[j] == 7:
            plt.scatter(latent_vecs_reduced[j, 0], latent_vecs_reduced[j, 1], marker='o', c='blue')
        elif y[j] == 8:
            plt.scatter(latent_vecs_reduced[j, 0], latent_vecs_reduced[j, 1], marker='o', c='gray')
        elif y[j] == 9:
            plt.scatter(latent_vecs_reduced[j, 0], latent_vecs_reduced[j, 1], marker='o', c='Tan')
        elif y[j] == 10:
            plt.scatter(latent_vecs_reduced[j, 0], latent_vecs_reduced[j, 1],marker='o', c='navy')
        elif y[j] == 11:
            plt.scatter(latent_vecs_reduced[j, 0], latent_vecs_reduced[j, 1],marker='o', c='bisque')
        elif y[j] == 12:
            plt.scatter(latent_vecs_reduced[j, 0], latent_vecs_reduced[j, 1], marker='o', c='Magenta')
        elif y[j] == 13:
            plt.scatter(latent_vecs_reduced[j, 0], latent_vecs_reduced[j, 1], marker='o', c='orange')
        elif y[j] == 14:
            plt.scatter(latent_vecs_reduced[j, 0], latent_vecs_reduced[j, 1], marker='o', c='darkviolet')
        elif y[j] == 15:
            plt.scatter(latent_vecs_reduced[j, 0], latent_vecs_reduced[j, 1], marker='o', c='khaki')
        elif y[j] == 16:
            plt.scatter(latent_vecs_reduced[j, 0], latent_vecs_reduced[j, 1], marker='o', c='lightgreen')
        else:
            plt.scatter(latent_vecs_reduced[j, 0], latent_vecs_reduced[j, 1],marker='o', c='lightgreen')
    cmap = mpl.colors.ListedColormap(['red', 'green', 'yellow', 'maroon', 'black', 'cyan', 'blue', 'gray', 'Tan',
                                      'navy', 'bisque', 'Magenta', 'orange', 'darkviolet', 'khaki', 'lightgreen'])
    # cmap = cmap.reversed()
    bounds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    plt.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap),
        boundaries=bounds,  # Adding values for extensions.
        ticks=bounds,
        spacing='proportional',
        orientation='vertical')
    plt.show()

t_sne(out, y)


