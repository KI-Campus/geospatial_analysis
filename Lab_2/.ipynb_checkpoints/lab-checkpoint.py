"""
Created on Sep 26 2018
Last edited on May 28 2024
@author: Dennis Wittich M.Sc. & Hubert Kanyamahanga, M.Sc.
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torchvision
import requests

# ========================= DATA GENERATION ============================================

def generate_gaussian_clusters(clusters, limits=(0, 255, 0, 255)):
    """Draws samples in 2D space according to the given normal distributions and limits.

        Parameters
        ----------
        clusters : list of tuples
            Each cluster is defined by 7 parameters:
            [(class, c_x, c_y, varx, vary, angle (degree), num_samples), ...]
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
    for (cx, cy, varx, vary, angle, c, num) in clusters:
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


# ========================= PRINTS / PLOTS ====================================

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

# this function creates "Data" folder it does not exist in the current home directory!

def create_data_folder():
    folder_name = "Data"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")


import requests

def download_file(url, save_path):
    """
    Downloads a file from the given URL and saves it to the specified location.

    Args:
        url (str): The URL of the file to download.
        save_path (str): The local path where the file will be saved.

    Raises:
        Exception: If the download fails.
    """
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        print(f"File successfully downloaded and saved to {save_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def download_data_from_seafile(url, save_path):
    """
    Downloads a file from the given URL and saves it to the specified location.

    Args:
        url (str): The URL of the file to download.
        save_path (str): The local path where the file will be saved.

    Raises:
        Exception: If the download fails.
    """
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Raise an error for bad responses (4xx or 5xx)
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        print(f"File successfully downloaded and saved to {save_path}")
    except Exception as e:
        print(f"An error occurred: {e}")



