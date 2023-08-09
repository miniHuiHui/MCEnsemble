import torch
from layers import SinkhornDistance
import numpy as np
import matplotlib.pyplot as plt
import cv2
#from scipy.stats import wasserstein_distance

'''
def cluster_norm(feature, label):
    xc = [feature[label == i] for i in np.unique(label)]
    cluster_center = [np.mean(x, axis = 0, keepdims = True) for x in xc]
   # print(cluster_center)
    feature_out = np.zeros_like(feature)
    for i in range(len(feature)):
        feature_out[i] = feature[i] - cluster_center[label[i]]
    
    return feature_out
'''
def cluster_center(feature,label):
    unique_labels = np.unique(label)
    cluster_center = np.array([np.mean(feature[label == i], axis=0) for i in unique_labels])
    return cluster_center

def cluster_norm(feature, label):
    unique_labels = np.unique(label)
    n_clusters = unique_labels.shape[0]
    cluster_center = np.array([np.mean(feature[label == i], axis=0) for i in unique_labels])
    print(cluster_center)
    feature_out = feature - cluster_center[label].reshape(-1, feature.shape[1])
    return feature_out

def sample_align_distance(feature1,feature2,p = 2):
    lp_dist = np.linalg.norm(feature1 - feature2, ord = p, axis = 1)
    #cosine_dist = 1.0 - np.dot(feature1,np.transpose(feature2))/(np.linalg.norm(feature1,axis = 1) * np.linalg.norm(feature2,axis = 1))
    return lp_dist#, cosine_dist

def normalization(feature):
    feature /= np.linalg.norm(feature, axis = -1, keepdims= True) + 1e-10
    m , s = np.mean(feature, axis = 0, keepdims = True), np.std(feature, axis = 0, keepdims = True)
    feature = (feature - m) / (s + 1e-10)
    return feature

def affine_transform(A, B):
    # convert matrices to homogeneous coordinates
    A_homo = np.hstack((A, np.ones((A.shape[0], 1))))
    B_homo = np.hstack((B, np.ones((B.shape[0], 1))))

    # solve for the affine transformation matrix
    T, _, _, _ = np.linalg.lstsq(A_homo, B_homo, rcond=None)

    return T.T


def rigid_transform_3D(A, B):
    assert len(A) == len(B)
    N = A.shape[0];
    mu_A = np.mean(A, axis=0)
    mu_B = np.mean(B, axis=0)

    AA = A - np.tile(mu_A, (N, 1))
    BB = B - np.tile(mu_B, (N, 1))
    H = np.transpose(AA).dot(BB)

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T.dot(U.T)

    if np.linalg.det(R) < 0:
        print ("Reflection detected")
        Vt[2, :] *= -1
        R = Vt.T.dot(U.T)

    t = -R.dot(mu_A.T) + mu_B.T
    
    print(R.shape)
    print(t.shape)

    return R, t


sl_features_train = np.load("./models/resnet18_cifar10/features1/features_train2.npy")
#print(sl_features_train)
ssl_features_train = np.load("./models/resnet18_cifar10/features3/features_train2.npy")
sl_labels_train = np.load("./models/resnet18_cifar10/features1/labels_train2.npy")
#ssl_labels_train = np.load("./features_all/self_supervised_cifar100/labels_test.npy")

sl_features_ood = np.load("./models/resnet18_cifar10/features1/features_ood.npy")
ssl_features_ood = np.load("./models/resnet18_cifar10/features3/features_ood.npy")
sl_labels_ood = np.load("./models/resnet18_cifar10/features1/labels_ood.npy")
print(sl_labels_train[:10])

sl_features_train = sl_features_train
ssl_features_train = ssl_features_train
sl_labels_train = sl_labels_train

sl_cluster_center = cluster_center(sl_features_train,sl_labels_train)
ssl_cluster_center = cluster_center(ssl_features_train,sl_labels_train)



sl_normal_feature = cluster_norm(sl_features_train,sl_labels_train)
ssl_normal_feature = cluster_norm(ssl_features_train,sl_labels_train)
print(np.mean(sl_normal_feature))
#sl_normal_feature = normalization(sl_features_train)
#ssl_normal_feature = normalization(ssl_features_train)
#print(ssl_normal_feature[0])

#R , t = rigid_transform_3D(sl_cluster_center,ssl_cluster_center)
R , t = rigid_transform_3D(sl_features_train[:30],ssl_features_train[:30])

#sl_features_2 = (R.dot(sl_features_train.T)) + np.tile(t,(len(ssl_features_train),1)).T
sl_features_2 = (R.dot(sl_features_ood.T)) + np.tile(t,(len(ssl_features_ood),1)).T
sl_features_2 = sl_features_2.T


x = torch.tensor(ssl_features_ood[30:130], dtype = torch.float32)
y = torch.tensor(sl_features_2[30:130], dtype = torch.float32)

sinkhorn = SinkhornDistance(eps=0.1, max_iter=10000, reduction='none', p = 2)
dist, P, C = sinkhorn(x,y)
plt.imshow(C)
#plt.title('Distance matrix')
plt.colorbar()
plt.show();
plt.imshow(P*100)
#plt.title('Coupling matrix')
plt.colorbar()
plt.show();

print("Sinkhorn distance: {:.3f}".format(dist.item()))

lp_dist = sample_align_distance(ssl_features_ood, sl_features_2, p = 2)





