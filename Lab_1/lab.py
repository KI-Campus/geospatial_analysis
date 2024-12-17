"""
Created on January 25, 2018
Last edited on May 7, 2024
@author: Dennis Wittich M.Sc. & Hubert Kanyamahanga, M.Sc.
"""

import os
import numpy as np
import PIL.Image
import IPython.display
import matplotlib
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture

import warnings
warnings.filterwarnings('ignore')

# ========================= PUBLIC METHODS ====================================

def normalize(I):
    """Normalizes a single channel image to the range 0.0 - 255.0.

        Parameters
        ----------
        I : ndarray of float64
            3D array representing the image to normalize

        Returns
        -------
        out : ndarray of float64
            3D array, normalized image

        Notes
        -----
        If minimum and maximum value in 'I' are identical, a copy of 'I' is returned.
    """

    min_value = np.min(I)
    max_value = np.max(I)

    if max_value == min_value:
        return np.copy(I)

    return (I - min_value) * 255 / (max_value - min_value)  # creates a copy

# ========================= IMAGE I/O =========================================

def imread3D(path):
    """Reads an image from disk. Returns the array representation.

        Parameters
        ----------
        path : str
            Path to file (including file extension)

        Returns
        -------
        out : ndarray of float64
            Image as 3D array

        Notes
        -----
        'I' will always have 3 dimensions: (rows, columns dimensions).
        Last dimension will be of length 1 or 3, depending on the image.

    """
    I = np.array(imageio.imread(path).astype(np.float64))  # first use imread() from imageio
    if I.ndim == 2:
        h, w = I.shape
        I = I.reshape((h, w, 1)).astype(np.float64)  # if image has two dimensions, we add one dimension
    else:
        if np.all(I[:, :, 0] == I[:, :, 1]) and np.all(I[:, :, 0] == I[:, :, 2]):
            return I[:, :, 0:1:].astype(np.float64)
        h, w, d = I.shape
        if d == 4:
            I = I[:, :, :3]  # if image has 3 dimensions and 4 channels, drop last channel

    return I

def imsave3D(path, I):
    """Saves the array representation of an image to disk.

        Parameters
        ----------
        path : str
            Path to file (including file extension)
        I : ndarray of float64
            Array representation of an image

        Returns
        -------
        out : none

        Notes
        -----
        The given array must have 3 dimensions,
        where the length of the last dimension is either 1 or 3.
    """
    assert I.ndim == 3, "image to save must have three dimensions!"
    h, w, d = I.shape
    assert d in {1, 3}, "depth of image to save must be 1 or 3!"
    I_uint8 = I.astype(np.uint8)
    if d == 1:
        I_uint8 = I_uint8.reshape(h, w)
    imageio.imsave(path, I_uint8)

def imshow3D(*I):
    """Shows the array representation of one or more images in a jupyter notebook.

        Parameters
        ----------
        I : ndarray of float64
            Array representation of an image
            Concatenates multiple images

        Returns
        -------
        out : none

        Notes
        -----
        The given array must have 3 dimensions,
        where the length of the last dimension is either 1 or 3.
    """

    if len(I) == 1:
        I = I[0]
    else:
        channels = [i.shape[2] for i in I]
        heights = [i.shape[0] for i in I]
        max_height = max(heights)
        max_channels = max(channels)

        if min(channels) != max_channels:  # if one image has three channels ..
            I = list(I)
            for i in range(len(I)):
                dim = channels[i]
                if dim == 1:  # .. and another has one channel ..
                    I[i] = np.dstack((I[i], I[i], I[i]))  # .. expand that image to three channels!

        if min(heights) != max_height:  # if heights of some images differ ..
            I = list(I)
            for i in range(len(I)):
                h, w, d = I[i].shape
                if h < max_height:  # .. expand by 'white' rows!
                    I_expanded = np.ones((max_height, w, d), dtype=np.float64) * 255
                    I_expanded[:h, :, :] = I[i]
                    I[i] = I_expanded

        seperator = np.ones((max_height, 3, max_channels), dtype=np.float64) * 255
        seperator[:, 1, :] *= 0
        I_sep = []
        for i in range(len(I)):
            I_sep.append(I[i])
            if i < (len(I) - 1):
                I_sep.append(seperator)
        I = np.hstack(I_sep)  # stack all images horizontally

    assert I.ndim == 3
    h, w, d = I.shape
    assert d in {1, 3}
    if d == 1:
        I = I.reshape(h, w)
    IPython.display.display(PIL.Image.fromarray(I.astype(np.ubyte)))

# ========================= DATA GENERATION ===================================

def generate_gaussian_clusters(clusters, limits=(0, 255, 0, 255)):
    """Draws samples in 2D space according to the given normal distributions and limits.

        Parameters
        ----------
        clusters : list of tuples
            Each cluster is defined by 7 parameters:
            [(class, c_x, c_y, angle (degree), varx, vary,  num_samples), ...]
        limits : tuple (optional)
            Contains min/max values for sample values:
            (min_x, max_x, min_x, max_x)

        Returns
        -------
        out : numpy.ndarray of type float32
            N x 3 matrix containing N samples (coordinates and label):
            [[x1, y1, c1], ..., [xN, yN, cN]]

        Notes
        -----
        Samples are drawn randomly (no fixed seed) and clipped to the given limits.
        Classes in result are also floats. Angles given in degree.
    """

    samples = []
    for (c, cx, cy, angle, varx, vary, num) in clusters:
        angle = angle / 180 * np.pi
        ca = np.cos(angle)
        sa = np.sin(angle)
        xr = np.random.randn((num)) * varx
        yr = np.random.randn((num)) * vary
        xs = ca * xr - sa * yr + cx
        ys = sa * xr + ca * yr + cy
        xs = np.minimum(np.maximum(xs, limits[0]), limits[1])
        ys = np.minimum(np.maximum(ys, limits[2]), limits[3])
        for x, y in zip(xs, ys):
            samples.append((x, y, c))

    return np.array(samples, dtype=float)

def get_feature_vectors_by_classes(X, y):
    """Groups samples by their classes.

        Parameters
        ----------
        X : Features as 2D array [num_samples x num_features]
        y : Labels as 1D array [num_samples]

        Returns
        -------
        out : list
            Element i of the list contains the samples
            of class i as 2D array [num_samples_i x num_features]

        Notes
        -----
        Samples are drawn randomly (no fixed seed) and clipped to the given limits.
        Classes in result are also floats. Angles given in rad.
    """

    num_classes = int(np.max(y) + 1)
    num_feature_vectors = len(X)
    feature_vectors_by_classes = []
    for c in range(num_classes):
        feature_vectors_by_classes.append(list())
    for i in range(num_feature_vectors):
        feature_vectors_by_classes[int(y[i])].append(X[i])
    for c in range(num_classes):
        feature_vectors_by_classes[c] = np.array(feature_vectors_by_classes[c])
    return feature_vectors_by_classes

# ========================= MODELS ============================================

class GaussianMixtureClassifier():
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters

    def fit(self, X, y):
        feature_vectors_by_classes = get_feature_vectors_by_classes(X, y)
        self.num_classes = len(feature_vectors_by_classes)

        if hasattr(self.num_clusters, '__len__'):
            assert len(self.num_clusters) == self.num_classes
        else:
            self.num_clusters = [self.num_clusters] * self.num_classes

        self.gmms = []
        for c in range(self.num_classes):
            self.gmms.append(GaussianMixture(n_components=self.num_clusters[c]))
            self.gmms[c].fit(feature_vectors_by_classes[c])

    def plot_sigma_ellipses(self, cmap, sigmas=[1]):
        for c in range(self.num_classes):
            gmm = self.gmms[c]
            num_clusters = gmm.means_.shape[0]
            for i in range(num_clusters):
                cov = gmm.covariances_[i]
                lambda_, v = np.linalg.eig(cov)
                lambda_ = np.sqrt(lambda_)
                for sigma in sigmas:
                    plt.gca().add_patch(
                        Ellipse(xy=gmm.means_[i], width=lambda_[0] * sigma * 2, height=lambda_[1] * sigma * 2,
                                angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])), facecolor='none', linewidth=3,
                                edgecolor=cmap.colors[int(c * (cmap.N - 1) / (self.num_classes - 1) + 0.5001)]))

    def uniform_prior(self):
        # Assumes uniform priors
        return [1/self.num_classes for i in range(self.num_classes)]
    
    def compute_likelihoods(self, X):
        num_samples = X.shape[0]
        likelihoods = np.zeros((num_samples, self.num_classes))
        for c in range(self.num_classes):
            likelihoods[:, c] = np.exp(self.gmms[c].score_samples(X))
        return likelihoods
    
    def likelihoods_to_posteriors(self, likelihoods, priors=None):
        # Computes posteriors for feature vectors
        #  X: feature_vectors [num_feature_vectors x num_features]
        
        if priors is None: priors = self.uniform_prior()
        assert np.sum(priors) == 1.0, "The prior has to sum up to one!"
        return likelihoods*priors/np.dot(likelihoods, np.asarray(priors).reshape(-1,1))
    
    def predict(self, X):
        # Predicts labels for feature vectors in X
        #  X: feature_vectors [num_feature_vectors x num_features]
        P = self.likelihoods_to_posteriors(self.compute_likelihoods(X))
        return np.argmax(P, axis=1)

# ========================= PRINTS / PLOTS ====================================

def plot_sigma_ellipses(ndc, cmap, sigmas=[1]):
    for c in range(ndc.num_classes):
        cov = ndc.covars[c]
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        for sigma in sigmas:
            plt.gca().add_patch(
                Ellipse(xy=ndc.means[c], width=lambda_[0] * sigma * 2, height=lambda_[1] * sigma * 2,
                        angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])), facecolor='none', linewidth=3,
                        edgecolor=cmap.colors[int(c * (cmap.N - 1) / (ndc.num_classes - 1) + 0.5001)]))

def print_probabilities(P, shape, text, n_cls):
    matplotlib.rcParams['figure.figsize'] = [p * 2 for p in matplotlib.rcParams['figure.figsize']]
    for i in range(n_cls):
        PROB_Class = P[:, i].reshape(shape)
        plt.subplot(1, n_cls, i + 1)
        plt.xlim((0, 255))
        plt.ylim((0, 255))
        plt.imshow(PROB_Class, cmap='gnuplot', vmin=0.0, vmax=np.max(P))
        plt.title(text + ' for Class ' + str(i))

    cax = plt.axes([0.15, 0.38, 0.7, 0.01])
    plt.colorbar(cax=cax, orientation='horizontal')
    plt.show()
    matplotlib.rcParams['figure.figsize'] = [p / 2 for p in matplotlib.rcParams['figure.figsize']]

def print_decision_boundaries(C, shape):
    predictions = C.reshape(shape)
    plt.xlim((0, 255))
    plt.ylim((0, 255))
    plt.imshow(predictions, cmap='Accent', vmin=0.0, vmax=np.max(C), interpolation='nearest')
    plt.title('Decision boundaries')
    cax = plt.axes([1.01, 0.15, 0.01, 0.7])
    plt.colorbar(cax=cax)
    plt.show()

def plot_patch(path):
    # Function to plot patch data from 'path'
    I_gt = imageio.imread(path + 'GT.png')
    I_irrg = imageio.imread(path + 'IR_R_G.png')
    I_ndsm = imageio.imread(path + 'NDSM.tif')
    I_ndsm -= np.min(I_ndsm)
    I_ndsm *= 255.0 / np.max(I_ndsm)
    I_ndsm = I_ndsm.astype(np.ubyte)
    I_ndsm = np.dstack((I_ndsm, I_ndsm, I_ndsm))

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(I_irrg)
    plt.title('NIR / R / G')
    plt.subplot(1, 3, 2)
    plt.imshow(I_ndsm)
    plt.title('NDSM')
    plt.subplot(1, 3, 3)
    plt.imshow(I_gt)
    plt.title('Ground Truth Labels')

def plot_pred_gt(pred, gt):
    # Function to plot prediction vs ground truth
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(pred)
    plt.title('Predicted Labels')
    plt.subplot(1, 2, 2)
    plt.imshow(gt)
    plt.title('Ground Truth Labels')

def feature_scale_to_0_255(feature_arr):
    feature_arr = ((feature_arr - feature_arr.min()) * (1/(feature_arr.max() - feature_arr.min()) * 255)).astype('uint8')
    return feature_arr


# this function creates "Data" folder it does not exist in the current home directory!

def create_data_folder():
    folder_name = "Data"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")

# Example usage:
# create_data_folder()