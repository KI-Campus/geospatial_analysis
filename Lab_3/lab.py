"""
Created on Sep 26 2018
Last edited on December 24 2024
@author: M.Sc. Dennis Wittich,  M.Sc. Hubert Kanyamahanga
"""

import numpy as np
import matplotlib.pyplot as plt
import torchvision
import pandas as pd
import torch, os
from torch.utils.data import Dataset
from torchvision.io import read_image
from os.path import join as pjoin
from functools import partial

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

colours = np.array(
    [[128, 0, 0],   # Building
    [128, 64, 128], # Road
    [0, 128, 0],    # Tree
    [128, 128, 0],  # Low vegetation
    [64, 64, 0],    # Human
    [64, 0, 128],   # Moving car
    [192, 0, 192],  # Static car
    [0, 0, 0],      # Background (clutter)
])

# ========================= DATA GENERATION ============================================

class MnistGenerator():

    def __init__(self, train_batch_size=32, num_train=59000, num_valid=1000, num_test=10000):
        self.trainset = torchvision.datasets.MNIST(root='./../mnist', train=True, download=True)
        self.testset = torchvision.datasets.MNIST(root='./../mnist', train=False, download=True)

        self.num_train = num_train
        self.num_valid = num_valid
        self.num_test = num_test

        assert num_train + num_valid <= len(self.trainset), "Not enough samples for training + validation"
        assert num_test <= len(self.testset), "Not enough samples for testing"

        self.Ti = 0
        self.TBS = train_batch_size

    def get_train_batch(self):
        Xs = np.zeros((self.TBS, 1, 28, 28), dtype=np.float32)
        Ys = np.zeros(self.TBS, dtype=np.int32)
        for i in range(self.TBS):
            Xi, Yi = self.trainset[self.Ti]
            Xs[i, 0] = np.array(Xi)
            Ys[i] = int(Yi)
            self.Ti += 1
            if self.Ti > self.num_train:
                self.Ti = 0
        return Xs, Ys

    def get_validation_batch(self):
        Xs = np.zeros((self.num_valid, 1, 28, 28), dtype=np.float32)
        Ys = np.zeros(self.num_valid, dtype=np.int32)
        for i in range(self.num_valid):
            Xi, Yi = self.trainset[self.num_train + i]
            Xs[i, 0] = np.array(Xi)
            Ys[i] = int(Yi)
            self.Ti += 1
            if self.Ti > self.num_train:
                self.Ti = 0
        return Xs, Ys

    def get_test_batch(self):
        Xs = np.zeros((self.num_test, 1, 28, 28), dtype=np.float32)
        Ys = np.zeros(self.num_test, dtype=np.int32)
        for i in range(self.num_test):
            Xi, Yi = self.testset[i]
            Xs[i, 0] = np.array(Xi)
            Ys[i] = int(Yi)
            self.Ti += 1
            if self.Ti > self.num_train:
                self.Ti = 0
        return Xs, Ys

# ========================= UAV DATA LOADER ====================================

class UaVidDataset(Dataset):
    """
        Implementation by inheritance from ``torch.util.data.Dataset`` 
        - init: Which data should be loaded? Setup attributes
        - len: How many items are in the dataset (defines length of loop)
        - getitem: Returns the item at index ``idx`` (may include preprocessing / augmentation steps)
    """
    def __init__(self, subset, tf = None):
        assert subset in ['train', 'val', 'test'], "Invalid subset"
        self.df = pd.read_pickle(f'./uavid_pkl/{subset}_df.pkl')
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 3]
        ref_path = self.df.iloc[idx, 2]
        image = (read_image(img_path) / 255.0)
        idmap = read_image(ref_path).long()
        sample = {'image':image, 'idmap':idmap}
        if self.tf:
            sample = self.tf(sample)
        return sample

class UaVidDataset_PL(Dataset):
    """
        Implementation by inheritance from ``torch.util.data.Dataset`` 
        - init: Which data should be loaded? Setup attributes
        - len: How many items are in the dataset (defines length of loop)
        - getitem: Returns the item at index ``idx`` (may include preprocessing / augmentation steps)
        - will preload 50 images and it's 3 times faster than UaVidDataset class.
    """
    def __init__(self, subset, num_smpl = 50):
        assert subset in ['train', 'val', 'test'], "Invalid subset"
        self.df = pd.read_pickle(f'./uavid_pkl/{subset}_df.pkl')
        self.num_smpl = num_smpl
        
        self.images = []
        self.refs = []
        print(f'preloading {num_smpl} images')
        
        for idx in range(num_smpl):
            img_path = self.df.iloc[idx, 3]
            ref_path = self.df.iloc[idx, 2]
            self.images.append((read_image(img_path)/255.0).type(torch.float16))
            self.refs.append(read_image(ref_path).type(torch.int8))
            
    def __len__(self):
        return self.num_smpl

    def __getitem__(self, idx):
        image = self.images[idx].float()
        idmap = self.refs[idx].long()
        return {'image': image, 'idmap': idmap}

def idmap2labelmap(idmap):
    """
    Function converts ID-maps to coloured label maps 
    """
    h,w = idmap.shape[:2]
    labelmap = colours[idmap.reshape(-1)].reshape((h,w,3))
    return labelmap

# ==================== SAVE and LOAD PARAMS ======================================

def save_net(net, name):
    save_dict = {'state_dict': net.state_dict()}
    torch.save(save_dict, pjoin('./checkpoints', f'{name}.pt'))

def load_net(net, name):
    load = torch.load if torch.cuda.is_available() else partial(torch.load, map_location='cpu')
    checkpoint = load(pjoin('./checkpoints', f'{name}.pt'))
    state_dict_to_load = checkpoint['state_dict']
    net.load_state_dict(state_dict_to_load)

# ========================= NETWORK EVALUATION ====================================

def eval_net(network, dataloader, metric='mf1'):
    '''The next function is used to evaluate on a subset (uses own 
    auxiliary functions to aggregate confusion matrix)'''
    
    num_cls = 7 
    ign_index = 7
    
    print('\nRunning validation .. ', end='')
    conf_matrix = np.zeros((num_cls, num_cls), int) 
    
    for batch in dataloader:
        image = batch['image'].to(device.type)
        idmap = batch['idmap']
        
        idmap[idmap==ign_index]=-1
        
        with torch.no_grad():                       # bit faster, steps for back propagation are skipped
            logits = network(image)
            preds = torch.argmax(logits, dim=1)
        preds_np = preds.cpu().data.numpy().ravel() # use Tensor.cpu() to convert Temsor to Numpy
        idmap_np = idmap.data.numpy().ravel()

        update_confusion_matrix(conf_matrix, preds_np, idmap_np)
        
    if metric == 'cm':
        return conf_matrix
    else:
        metrics = get_confusion_metrics(conf_matrix)
        return metrics[metric]       

def update_confusion_matrix(confusions, predicted_labels, reference_labels):
    # reference labels with label < 0 will not be considered
    reshaped_pr = np.ravel(predicted_labels)
    reshaped_gt = np.ravel(reference_labels)
    for predicted, actual in zip(reshaped_pr, reshaped_gt):
        if actual >= 0 and predicted >= 0:
            confusions[predicted, actual] += 1

def get_confusion_metrics(confusion_matrix):
    """Computes confusion metrics out of a confusion matrix (N classes)
        Parameters
        ----------
        confusion_matrix : numpy.ndarray
            Confusion matrix [N x N]
        Returns
        -------
        metrics : dict
            a dictionary holding all computed metrics
        Notes
        -----
        Metrics are: 'percentages', 'precisions', 'recalls', 'f1s', 'mf1', 'oa'
    """

    tp = np.diag(confusion_matrix)
    tp_fn = np.sum(confusion_matrix, axis=0)
    tp_fp = np.sum(confusion_matrix, axis=1)

    has_no_rp = tp_fn == 0
    has_no_pp = tp_fp == 0

    tp_fn[has_no_rp] = 1
    tp_fp[has_no_pp] = 1

    percentages = tp_fn / np.sum(confusion_matrix)
    precisions = tp / tp_fp
    recalls = tp / tp_fn

    p_zero = precisions == 0
    precisions[p_zero] = 1

    f1s = 2 * (precisions * recalls) / (precisions + recalls)
    ious = tp / (tp_fn + tp_fp - tp)

    precisions[has_no_pp] *= 0.0
    precisions[p_zero] *= 0.0
    recalls[has_no_rp] *= 0.0

    f1s[p_zero] *= 0.0
    f1s[percentages == 0.0] = np.nan
    ious[percentages == 0.0] = np.nan

    mf1 = np.nanmean(f1s)
    miou = np.nanmean(ious)
    oa = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    metrics = {'percentages': percentages,
               'precisions': precisions,
               'recalls': recalls,
               'f1s': f1s,
               'mf1': mf1,
               'ious': ious,
               'miou': miou,
               'oa': oa}

    return metrics

def print_metrics(confusions):
    metrics = get_confusion_metrics(confusions)

    print('\nclass | pct of data | precision |   recall  |    f1     |    iou',
          '\n-----------------------------------------------------------------')

    percentages = metrics["percentages"]
    precisions = metrics["precisions"]
    recall = metrics["recalls"]
    f1 = metrics["f1s"]
    ious = metrics["ious"]
    mf1 = metrics["mf1"]
    miou = metrics["miou"]
    oa = metrics["oa"]

    for i in range(len(percentages)):
        pct = '{:.3%}'.format(percentages[i]).rjust(9)
        p = '{:.3%}'.format(precisions[i]).rjust(7)
        r = '{:.3%}'.format(recall[i]).rjust(7)
        f = '{:.3%}'.format(f1[i]).rjust(7)
        u = '{:.3%}'.format(ious[i]).rjust(7)
        print('   {:2d} |  {}  |  {}  |  {}  |  {}  |  {}\n'.format(i, pct, p, r, f, u))

    print('mean f1-score: {:.3%}'.format(mf1))
    print('mean iou: {:.3%}'.format(miou))
    print('Overall accuracy: {:.3%}'.format(oa))
    print('Samples: {}'.format(np.sum(confusions)))

def print_summary(Ls, TBAs, VAs, CM):
    print('\nFinal validation accuracy: {:.1%}'.format(VAs[-1]))
    print('\nTEST SET ACCURACY: {:.1%}\n'.format(np.trace(CM)/np.sum(CM)))

    plt.subplot(1,2,1)
    plt.plot(Ls,label='Cross Entropy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot([x*100 for x in TBAs], label='Training (Batch) Accuracy')
    plt.plot([x*100 for x in VAs], label='Validation Accuracy')
    plt.ylim(0, 105)
    plt.legend()
    plt.show()
    

    import os

# this function creates "checkpoints" folder it does not exist in the current home directory!

def create_checkpoints_folder():
    folder_name = "checkpoints"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")

# Example usage:
# create_checkpoints_folder()