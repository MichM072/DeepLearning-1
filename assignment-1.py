# %% Cell 1
import gzip
import math as m
import os
import pickle
import random as r
from urllib import request

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# %% Cell 2
class simple_NN:
    def __init__(self) -> None:
        self.x = [1.0, -1.0]
        self.t = [1.0, 0.0]
        self.k = [0.0, 0.0, 0.0]
        self.h = [0.0, 0.0, 0.0]
        self.w = [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]
        self.v = [[1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0]]
        self.b = [0.0, 0.0, 0.0]
        self.c = [0.0, 0.0]
        self.o = [0.0, 0.0]
        self.y = [0.0, 0.0]
        self.L = 0.0

        # Backpropagation
        self.d_y = [0.0, 0.0]
        self.d_o = [0.0, 0.0]
        self.d_v = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        self.d_h = [0.0, 0.0, 0.0]
        self.d_c = [0.0, 0.0]
        self.d_k = [0.0, 0.0, 0.0]
        self.d_w = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        self.d_b = [0.0, 0.0, 0.0]

    def initialize_weights(self, strategy="random") -> None:
        # TODO: Ask ta if this random is correct!
        if strategy == "random":
            for j in range(len(self.k)):
                for i in range(len(self.x)):
                    self.w[i][j] = r.uniform(-1.0, 1.0)

            for j in range(len(self.o)):
                for i in range(len(self.h)):
                    self.v[i][j] = r.uniform(-1.0, 1.0)

    def print_grads(self) -> None:
        print("Gradients:")
        gradient_list = ["y", "o", "v", "h", "c", "k", "w", "b"]
        gradient_values = [
            self.d_y,
            self.d_o,
            self.d_v,
            self.d_h,
            self.d_c,
            self.d_k,
            self.d_w,
            self.d_b,
        ]
        for name, value in zip(gradient_list, gradient_values):
            print(f"{name}: {value}")

    @staticmethod
    def sigmoid(x) -> float:
        return 1 / (1 + m.exp(-x))

    @staticmethod
    def softmax(x) -> list[float]:
        exp_x = [m.exp(xi) for xi in x]
        sum_exp_x = sum(exp_x)
        return [xi / sum_exp_x for xi in exp_x]

    def forward(self, x, t) -> None:
        self.x = x
        self.t = t

        for j in range(len(self.k)):
            self.k[j] = self.b[j]
            for i in range(len(self.x)):
                self.k[j] += self.w[i][j] * self.x[i]

        for i in range(len(self.k)):
            self.h[i] = self.sigmoid(self.k[i])

        for i in range(len(self.o)):
            self.o[i] = self.c[i]
            for j in range(len(self.h)):
                self.o[i] += self.v[j][i] * self.h[j]

        self.y = self.softmax(self.o)

        # TODO: Check if this is correct.
        self.L = sum(-m.log(y) * t for y, t in zip(self.y, self.t))

    def backward(self) -> None:
        # TODO: Do we really need d_y? Ask ta!
        self.d_y = [(-1 / y) * t for y, t in zip(self.y, self.t)]
        self.d_o = [y - t for y, t in zip(self.y, self.t)]

        for j in range(len(self.h)):
            sum_grad_h = 0.0
            for i in range(len(self.d_o)):
                self.d_v[j][i] = self.d_o[i] * self.h[j]
                sum_grad_h += self.d_o[i] * self.v[j][i]
            self.d_h[j] = sum_grad_h

        self.d_c = self.d_o
        self.d_k = [dh * h * (1 - h) for dh, h in zip(self.d_h, self.h)]

        for j in range(len(self.d_k)):
            for i in range(len(self.x)):
                self.d_w[i][j] = self.d_k[j] * self.x[i]
            self.d_b[j] = self.d_k[j]

    def update(self, lr) -> None:
        for j in range(len(self.k)):
            for i in range(len(self.x)):
                self.w[i][j] = self.w[i][j] - lr * self.d_w[i][j]

        for j in range(len(self.o)):
            for i in range(len(self.h)):
                self.v[i][j] = self.v[i][j] - lr * self.d_v[i][j]

        for i in range(len(self.b)):
            self.b[i] = self.b[i] - lr * self.d_b[i]

        for i in range(len(self.c)):
            self.c[i] = self.c[i] - lr * self.d_c[i]

    def train(self, xtrain, ytrain, xeval, yeval, epochs=10, lr=0.02) -> tuple:
        loss_history = []
        loss_history_eval = []
        for epoch in tqdm(range(epochs), desc="Epochs"):
            epoch_loss = 0.0
            for xtrain_i, ytrain_i in zip(xtrain, ytrain):
                self.forward(xtrain_i, ytrain_i)
                self.backward()
                self.update(lr)

            # Now compute the loss over trained network within epoch
            for xtrain_i, ytrain_i in zip(xtrain, ytrain):
                self.forward(xtrain_i, ytrain_i)
                epoch_loss += self.L

            avg_loss = epoch_loss / len(xtrain)
            print(f"Epoch {epoch} avg_loss_training: {avg_loss}")
            loss_history.append(avg_loss)

            # Now evaluate on the evaluation set
            epoch_loss_eval = 0.0
            for xeval_i, yeval_i in zip(xeval, yeval):
                self.forward(xeval_i, yeval_i)
                epoch_loss_eval += self.L

            avg_loss_eval = epoch_loss_eval / len(xeval)
            print(f"Epoch {epoch} avg_loss_eval: {avg_loss_eval}")
            loss_history_eval.append(avg_loss_eval)

        return loss_history, loss_history_eval


# %% Cell 3

# Ask TA if output for w is correct
input = [1.0, -1.0]
target = [1, 0]
test_NN = simple_NN()

test_NN.forward(input, target)
test_NN.backward()

test_NN.print_grads()

# %% Cell 4
# -- assignment 1 --
# Code copied from: https://gist.github.com/pbloem/bd8348d58251872d9ca10de4816945e4
# Imports moved to top for visual clarity!


def load_synth(num_train=60_000, num_val=10_000, seed=0):
    """
    Load some very basic synthetic data that should be easy to classify. Two features, so that we can plot the
    decision boundary (which is an ellipse in the feature space).

    :param num_train: Number of training instances
    :param num_val: Number of test/validation instances
    :param num_features: Number of features per instance

    :return: Two tuples and an integer: (xtrain, ytrain), (xval, yval), num_cls. The first contains a matrix of training
     data with 2 features as a numpy floating point array, and the corresponding classification labels as a numpy
     integer array. The second contains the test/validation data in the same format. The last integer contains the
     number of classes (this is always 2 for this function).
    """
    np.random.seed(seed)

    THRESHOLD = 0.6
    quad = np.asarray([[1, -0.05], [1, 0.4]])

    ntotal = num_train + num_val

    x = np.random.randn(ntotal, 2)

    # compute the quadratic form
    q = np.einsum("bf, fk, bk -> b", x, quad, x)
    y = (q > THRESHOLD).astype(int)

    return (x[:num_train, :], y[:num_train]), (x[num_train:, :], y[num_train:]), 2


def load_mnist(final=False, flatten=True, shuffle_seed=0):
    """
    Load the MNIST data.

    :param final: If true, return the canonical test/train split. If false, split some validation data from the training
       data and keep the test data hidden.
    :param flatten: If true, each instance is flattened into a vector, so that the data is returns as a matrix with 768
        columns. If false, the data is returned as a 3-tensor preserving each image as a matrix.
    :param shuffle_seed If >= 0, the data is shuffled. This keeps the canonical test/train split, but shuffles each
        internally before splitting off a validation set. The given number is used as a seed. Note that the original data
        is _not_ shuffled, but ordered by writer. This means that there will be a distribution shift between train and val
        if the data is not shuffled.

    :return: Two tuples and an integer: (xtrain, ytrain), (xval, yval), num_cls. The first contains a matrix of training
     data and the corresponding classification labels as a numpy integer array. The second contains the test/validation
     data in the same format. The last integer contains the number of classes (this is always 2 for this function).

    """

    if not os.path.isfile("mnist.pkl"):
        init()

    xtrain, ytrain, xtest, ytest = load()
    xtl, xsl = xtrain.shape[0], xtest.shape[0]

    if flatten:
        xtrain = xtrain.reshape(xtl, -1)
        xtest = xtest.reshape(xsl, -1)

    if shuffle_seed >= 0:
        rng = np.random.default_rng(shuffle_seed)

        p = rng.permutation(xtrain.shape[0])
        xtrain, ytrain = xtrain[p], ytrain[p]

        p = rng.permutation(xtest.shape[0])
        xtest, ytest = xtest[p], ytest[p]

    if not final:  # return the flattened images
        return (xtrain[:-5000], ytrain[:-5000]), (xtrain[-5000:], ytrain[-5000:]), 10

    return (xtrain, ytrain), (xtest, ytest), 10


# Numpy-only MNIST loader. Courtesy of Hyeonseok Jung
# https://github.com/hsjeong5/MNIST-for-Numpy

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"],
]


def download_mnist():
    base_url = (
        "https://peterbloem.nl/files/mnist/"  # "http://yann.lecun.com/exdb/mnist/"
    )
    for name in filename:
        print("Downloading " + name[1] + "...")
        request.urlretrieve(base_url + name[1], name[1])
    print("Download complete.")


def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], "rb") as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(
                -1, 28 * 28
            )
    for name in filename[-2:]:
        with gzip.open(name[1], "rb") as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", "wb") as f:
        pickle.dump(mnist, f)
    print("Save complete.")


def init():
    download_mnist()
    save_mnist()


def load():
    with open("mnist.pkl", "rb") as f:
        mnist = pickle.load(f)
    return (
        mnist["training_images"],
        mnist["training_labels"],
        mnist["test_images"],
        mnist["test_labels"],
    )


# %% Cell 5
(xtrain, ytrain), (xval, yval), num_cls = load_synth()


# %% Cell 6


def encode_labels(labels, num_classes) -> list[list]:
    one_hot_labels = []
    for label in labels:
        vec = [0] * num_classes
        vec[label] = 1
        one_hot_labels.append(vec)
    return one_hot_labels


def normalize_values(train, val, range: tuple) -> tuple:
    train_min = float("inf")
    train_max = float("-inf")

    for row in train:
        for i in row:
            if i < train_min:
                train_min = i
            elif i > train_max:
                train_max = i

    normalized_train = []
    normalized_val = []

    for mode, norm_list in zip([train, val], [normalized_train, normalized_val]):
        for row in mode:
            row_norm = []
            for i in row:
                train_i = (range[1] - range[0]) * (
                    (i - train_min) / (train_max - train_min)
                ) + range[0]
                row_norm.append(train_i)

            norm_list.append(row_norm)

    # for row_train, row_val in zip(train, val):
    #     row_train_norm = []
    #     row_val_norm = []
    #     for i, j in zip(row_train, row_val):
    #         train_i = (range[1] - range[0]) * (
    #             (i - train_min) / (train_max - train_min)
    #         ) + range[0]

    #         val_j = (range[1] - range[0]) * (
    #             (j - train_min) / (train_max - train_min)
    #         ) + range[0]

    #         row_train_norm.append(train_i)
    #         row_val_norm.append(val_j)

    #     normalized_train.append(row_train_norm)
    #     normalized_val.append(row_val_norm)

    return normalized_train, normalized_val


# First transform to list since we are using plain python

xtrain = xtrain.tolist() if not isinstance(xtrain, list) else xtrain
xval = xval.tolist() if not isinstance(xval, list) else xval

print("Before norm:")
print(f"xtrain: {xtrain[:10]}")
print(f"xval: {xval[:10]}")

xtrain_norm, xval_norm = normalize_values(xtrain, xval, (-1, 1))
ytrain_enc = encode_labels(ytrain, num_cls)
yval_enc = encode_labels(yval, num_cls)

print("After norm:")
print(f"xtrain: {xtrain_norm[:10]}")
print(f"xval: {xval_norm[:10]}")

# %% Cell 7
neural_network_q4 = simple_NN()
loss_train, loss_eval = neural_network_q4.train(
    xtrain_norm, ytrain_enc, xval_norm, yval_enc, epochs=50, lr=0.02
)

# %% Cell 8
plt.figure(figsize=(15, 10))
x = range(len(loss_train))
plt.plot(x, loss_train, label="Train")
# plt.plot(x, loss_eval, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend()

plt.show()


# %% Cell 9
class vectorizedNN:
    def __init__(self) -> None:
        # Forward
        self.b = None
        self.w = None
        self.k = None
        self.h = None
        self.v = None
        self.y = np.zeros(10)
        self.c = None
        self.L = 0.0
        self.layers = []

        # Backward
        self.d_y = None
        self.d_o = None
        self.d_v = None
        self.d_h = None
        self.d_c = None
        self.d_k = None
        self.d_w = None
        self.d_b = None

    def build_layer_1(self, input_size, output_size):
        if self.w is None:
            self.w = np.random.randn(output_size, input_size)

        if self.b is None:
            self.b = np.zeros(output_size)

    def build_layer_2(self, input_size, output_size):
        if self.v is None:
            self.v = np.random.randn(output_size, input_size)

        if self.c is None:
            self.c = np.zeros(output_size)

    def sigmoid(self, k):
        exp_k = np.exp(k - np.max(k))
        return 1 / (1 + np.exp(-exp_k))

    def softmax(self, o):
        exp_o = np.exp(o - np.max(o, axis=0, keepdims=True))
        return exp_o / np.sum(exp_o, axis=0, keepdims=True)

    def forward(self, x, t):
        self.k = self.w @ x + self.b
        self.h = self.sigmoid(self.k)
        self.o = self.v @ self.h + self.c
        self.y = self.softmax(self.o)
        self.L = -np.log(self.y[t])


# %% Cell 10
def vectorized_normalization(train, val, range: tuple) -> tuple:
    train_min = np.min(train)
    train_max = np.max(train)

    normalized_train = (range[1] - range[0]) * (
        (train - train_min) / (train_max - train_min)
    ) + range[0]

    normalized_val = (range[1] - range[0]) * (
        (val - train_min) / (train_max - train_min)
    ) + range[0]

    return normalized_train, normalized_val


# %% Cell 11
(xtrain_mnist, ytrain_mnist), (xval_mnist, yval_mnist), num_cls_mnist = load_mnist()
# %% Cell 12
print(f"xtrain shape: {np.array(xtrain_mnist).shape}")
img_shape = xtrain_mnist.shape[1]
print(f"ytrain: {ytrain_mnist[:10]}")
# %% Cell test 2
print(
    f"Before norm:{np.max(xtrain_mnist), np.min(xtrain_mnist)}, {np.max(xval_mnist), np.min(xval_mnist)}"
)
xtrain_mnist_norm, xval_mnist_norm = vectorized_normalization(
    np.array(xtrain_mnist), np.array(xval_mnist), (0, 1)
)
print(
    f"After norm:{np.max(xtrain_mnist_norm), np.min(xtrain_mnist_norm)}, {np.max(xval_mnist_norm), np.min(xval_mnist_norm)}"
)


# %% Cell test
test_NN = vectorizedNN()
test_NN.build_layer_1(img_shape, 300)
test_NN.build_layer_2(300, num_cls_mnist)
test_NN.forward(xtrain_mnist_norm[0], ytrain_mnist[0])
print(test_NN.L)
