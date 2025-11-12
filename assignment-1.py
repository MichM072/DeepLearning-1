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
def plot_loss(loss, save_img=False, img_title="loss_plot") -> None:
    plt.figure(figsize=(15, 10))
    x = range(len(loss))
    plt.plot(x, loss, label="Train")
    plt.tick_params(axis="both", which="major", labelsize=14)
    # plt.plot(x, loss_eval, label="Validation")
    plt.xlabel("Epoch", fontsize=14, fontweight="bold")
    plt.ylabel("Loss", fontsize=14, fontweight="bold")
    plt.grid()
    plt.legend()
    plt.tight_layout()

    if save_img:
        if not os.path.exists("./images"):
            os.makedirs("./images")
        plt.savefig(f"./images/{img_title}.png")

    plt.show()


try:
    plot_loss(loss_train)
except Exception as e:
    print(e)


# %% Cell 9
class vectorizedNN:
    def __init__(self) -> None:
        # Forward
        self.b = np.empty(0)
        self.w = np.empty(0)
        self.k = np.empty(0)
        self.h = np.empty(0)
        self.v = np.empty(0)
        self.y = np.zeros(10)
        self.c = np.empty(0)
        self.L = 0.0
        self.layers = []

        # Backward
        self.d_y = np.empty(0)
        self.d_o = np.empty(0)
        self.d_v = np.empty(0)
        self.d_h = np.empty(0)
        self.d_c = np.empty(0)
        self.d_k = np.empty(0)
        self.d_w = np.empty(0)
        self.d_b = np.empty(0)

        # Batch save
        self.batch_d_w = np.empty(0)
        self.batch_d_b = np.empty(0)
        self.batch_d_v = np.empty(0)
        self.batch_d_c = np.empty(0)
        self.batch_size = 0

    def save_batch_grads(self) -> None:
        self.batch_d_w += self.d_w
        self.batch_d_b += self.d_b
        self.batch_d_v += self.d_v
        self.batch_d_c += self.d_c
        self.batch_size += 1

    def reset_batch_grads(self) -> None:
        self.batch_d_w = np.zeros_like(self.w)
        self.batch_d_b = np.zeros_like(self.b)
        self.batch_d_v = np.zeros_like(self.v)
        self.batch_d_c = np.zeros_like(self.c)
        self.batch_size = 0

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

    def build_layer_1(self, input_size, output_size) -> None:
        if self.w.size == 0:
            self.w = np.random.normal(0.0, 0.2, size=(output_size, input_size[0]))
            self.batch_d_w = np.zeros((output_size, input_size[0]))

        if self.b.size == 0:
            self.b = np.zeros((output_size, 1))
            self.batch_d_b = np.zeros((output_size, 1))

    def build_layer_2(self, input_size, output_size) -> None:
        if self.v.size == 0:
            self.v = np.random.normal(0.0, 0.2, size=(output_size, input_size))
            self.batch_d_v = np.zeros((output_size, input_size))

        if self.c.size == 0:
            self.c = np.zeros((output_size, 1))
            self.batch_d_c = np.zeros((output_size, 1))

    def sigmoid(self, k) -> np.ndarray:
        # Prevents overflow
        # Source: https://blog.dailydoseofds.com/p/a-highly-overlooked-point-in-the
        sig = np.where(k > 0, 1 / (1 + np.exp(-k)), np.exp(k) / (1 + np.exp(k)))
        return sig

    def softmax(self, o) -> np.ndarray:
        exp_o = np.exp(o - np.max(o, axis=0, keepdims=True))
        # exp_o = np.exp(o)
        return exp_o / np.sum(exp_o, axis=0, keepdims=True)

    def forward(self, x, t) -> None:
        self.k = self.w @ x + self.b
        self.h = self.sigmoid(self.k)
        self.o = self.v @ self.h + self.c
        self.y = self.softmax(self.o)
        self.L = -np.log(self.y[t].item())

    def backward(self, x, t) -> None:
        # Create one-hot vector for target as 10 classes are reasonably small
        ot = np.zeros_like(self.y)
        ot[t] = 1.0
        self.d_y = -(1 / self.y[t])
        self.d_o = self.y - ot
        self.d_v = self.d_o @ self.h.T
        self.d_h = self.v.T @ self.d_o
        self.d_c = self.d_o
        self.d_k = self.d_h * self.h * (1 - self.h)
        self.d_w = self.d_k @ x.T
        self.d_b = self.d_k

    def update(self, lr) -> None:
        self.w -= lr * (1 / self.batch_size) * self.batch_d_w
        self.v -= lr * (1 / self.batch_size) * self.batch_d_v
        self.b -= lr * (1 / self.batch_size) * self.batch_d_b
        self.c -= lr * (1 / self.batch_size) * self.batch_d_c

    def train(
        self, xtrain, ytrain, xval, yval, minibatch_size, epochs=10, lr=0.02
    ) -> list:
        epoch_loss_history = []
        for epoch in tqdm(range(epochs), desc="Epochs"):
            epoch_loss = 0.0
            i = 0
            for xtrain_i, ytrain_i in zip(xtrain, ytrain):
                xtrain_i = xtrain_i.reshape(-1, 1)
                self.forward(xtrain_i, ytrain_i)
                self.backward(xtrain_i, ytrain_i)
                self.save_batch_grads()

                assert np.sum(self.y) <= 1.1, f"y does not sum to 1: {np.sum(self.y)}"

                if (i + 1) % minibatch_size == 0:
                    # TODO: Check if this is correct!
                    self.update(lr)
                    self.reset_batch_grads()
                i += 1

            if self.batch_size != 0:
                self.update(lr)
                self.reset_batch_grads()

            for xtrain_i, ytrain_i in zip(xtrain, ytrain):
                xtrain_i = xtrain_i.reshape(-1, 1)
                self.forward(xtrain_i, ytrain_i)
                epoch_loss += self.L

            epoch_loss = epoch_loss / len(xtrain)
            epoch_loss_history.append(epoch_loss)
            print(f"Train loss epoch {epoch}: {epoch_loss}")

        return epoch_loss_history


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
img_shape = xtrain_mnist.shape[1:]
print(f"ytrain: {ytrain_mnist[:10]}")
# %% Cell test 13
print(
    f"Before norm:{np.max(xtrain_mnist), np.min(xtrain_mnist)}, {np.max(xval_mnist), np.min(xval_mnist)}"
)
xtrain_mnist_norm, xval_mnist_norm = vectorized_normalization(
    np.array(xtrain_mnist), np.array(xval_mnist), (0, 1)
)
print(
    f"After norm:{np.max(xtrain_mnist_norm), np.min(xtrain_mnist_norm)}, {np.max(xval_mnist_norm), np.min(xval_mnist_norm)}"
)

# %% Cell 14
vec_NN = vectorizedNN()
vec_NN.build_layer_1(img_shape, 300)
vec_NN.build_layer_2(300, num_cls_mnist)
vec_loss = vec_NN.train(
    xtrain_mnist_norm,
    ytrain_mnist,
    xval_mnist_norm,
    yval_mnist,
    minibatch_size=500,
    epochs=10,
    lr=0.02,
)

# %% Cell 15
plot_loss(vec_loss, save_img=True, img_title="vec_loss")


# %% Cell 16
class batched_vectorizedNN:
    def __init__(self) -> None:
        # Forward
        self.b = np.empty(0)
        self.w = np.empty(0)
        self.k = np.empty(0)
        self.h = np.empty(0)
        self.v = np.empty(0)
        self.y = np.zeros(10)
        self.c = np.empty(0)
        self.L = 0.0
        self.layers = []

        # Backward
        self.d_y = np.empty(0)
        self.d_o = np.empty(0)
        self.d_v = np.empty(0)
        self.d_h = np.empty(0)
        self.d_c = np.empty(0)
        self.d_k = np.empty(0)
        self.d_w = np.empty(0)
        self.d_b = np.empty(0)
        self.batch_size = 0

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

    def build_layer_1(self, input_size, output_size) -> None:
        if self.w.size == 0:
            self.w = np.random.normal(0.0, 0.2, size=(output_size, input_size[0]))
            self.batch_d_w = np.zeros((output_size, input_size[0]))

        if self.b.size == 0:
            self.b = np.zeros((output_size, 1))
            self.batch_d_b = np.zeros((output_size, 1))

    def build_layer_2(self, input_size, output_size) -> None:
        if self.v.size == 0:
            self.v = np.random.normal(0.0, 0.2, size=(output_size, input_size))
            self.batch_d_v = np.zeros((output_size, input_size))

        if self.c.size == 0:
            self.c = np.zeros((output_size, 1))
            self.batch_d_c = np.zeros((output_size, 1))

    def sigmoid(self, k) -> np.ndarray:
        # Prevents overflow
        # Source: https://blog.dailydoseofds.com/p/a-highly-overlooked-point-in-the
        sig = np.where(k > 0, 1 / (1 + np.exp(-k)), np.exp(k) / (1 + np.exp(k)))
        return sig

    def softmax(self, o) -> np.ndarray:
        exp_o = np.exp(o - np.max(o, axis=0, keepdims=True))
        # exp_o = np.exp(o)
        return exp_o / np.sum(exp_o, axis=0, keepdims=True)

    def forward(self, x, t) -> None:
        self.batch_size = x.shape[1]
        self.k = self.w @ x + self.b
        self.h = self.sigmoid(self.k)
        self.o = self.v @ self.h + self.c
        self.y = self.softmax(self.o)

        batched_y = self.y[t, np.arange(self.batch_size)]

        self.L = np.sum(-np.log(batched_y)) / self.batch_size

    def backward(self, x, t) -> None:
        self.d_y = -(1 / self.y[t, np.arange(self.batch_size)])
        # TODO: Check if this worked properly
        self.d_o = self.y.copy()
        self.d_o[t, np.arange(self.batch_size)] -= 1.0
        self.d_v = self.d_o @ self.h.T
        self.d_h = self.v.T @ self.d_o
        self.d_c = np.sum(self.d_o, axis=1, keepdims=True)
        self.d_k = self.d_h * self.h * (1 - self.h)
        self.d_w = self.d_k @ x.T
        self.d_b = np.sum(self.d_k, axis=1, keepdims=True)

    def update(self, lr) -> None:
        self.w -= lr * (1 / self.batch_size) * self.d_w
        self.v -= lr * (1 / self.batch_size) * self.d_v
        self.b -= lr * (1 / self.batch_size) * self.d_b
        self.c -= lr * (1 / self.batch_size) * self.d_c

    def train(
        self, xtrain, ytrain, xval, yval, minibatch_size, epochs=10, lr=0.02
    ) -> tuple:
        epoch_loss_history = []
        epoch_loss_history_val = []
        for epoch in tqdm(range(epochs), desc="Epochs"):
            epoch_loss = 0.0
            for slice in range(0, len(xtrain), minibatch_size):
                slice_end = min(slice + minibatch_size, len(xtrain))
                xtrain_batch = xtrain[slice:slice_end].T
                ytrain_batch = ytrain[slice:slice_end]
                self.forward(xtrain_batch, ytrain_batch)
                self.backward(xtrain_batch, ytrain_batch)
                self.update(lr)

            # Training loss:
            num_batches = 0
            for slice in range(0, len(xtrain), minibatch_size):
                slice_end = min(slice + minibatch_size, len(xtrain))
                xtrain_batch = xtrain[slice:slice_end].T
                ytrain_batch = ytrain[slice:slice_end]
                self.forward(xtrain_batch, ytrain_batch)

                epoch_loss += self.L
                num_batches += 1

            epoch_loss = epoch_loss / num_batches
            epoch_loss_history.append(epoch_loss)
            print(f"Train loss epoch {epoch}: {epoch_loss}")

            val_epoch_loss = 0.0
            num_batches_val = 0
            for slice in range(0, len(xval), minibatch_size):
                slice_end = min(slice + minibatch_size, len(xval))
                xval_batch = xval[slice:slice_end].T
                yval_batch = yval[slice:slice_end]
                self.forward(xval_batch, yval_batch)

                val_epoch_loss += self.L
                num_batches_val += 1

            val_epoch_loss = val_epoch_loss / num_batches_val
            epoch_loss_history_val.append(val_epoch_loss)
            print(f"Validation loss epoch {epoch}: {val_epoch_loss}")

        return epoch_loss_history, epoch_loss_history_val


# %% Cell 17
batched_vec_NN = batched_vectorizedNN()
batched_vec_NN.build_layer_1(img_shape, 300)
batched_vec_NN.build_layer_2(300, num_cls_mnist)
batched_loss, _ = batched_vec_NN.train(
    xtrain_mnist_norm,
    ytrain_mnist,
    xval_mnist_norm,
    yval_mnist,
    minibatch_size=500,
    epochs=50,
    lr=0.05,
)

# %% Cell 18
plot_loss(batched_loss, save_img=True, img_title="batched_vec_loss")
# %% Cell 19