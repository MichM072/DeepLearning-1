# %% Cell 1
import math as m


# %% Cell 2
class simple_NN:
    def __init__(self, input_x: list, target: list) -> None:
        self.x = input_x
        self.t = target
        self.k = [0.0, 0.0, 0.0]
        self.h = [0.0, 0.0, 0.0]
        self.w = [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]
        self.v = [[1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0]]
        self.bias_x = [0.0, 0.0, 0.0]
        self.bias_c = [0.0, 0.0]
        self.o = [0.0, 0.0]
        self.y = [0.0, 0.0]
        self.L = 0.0

        # Backpropagation
        self.d_o = [0.0, 0.0]
        self.d_y = [0.0, 0.0]
        self.d_v = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        self.d_h = [0.0, 0.0, 0.0]
        self.d_w = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        self.d_k = [0.0, 0.0, 0.0]
        self.d_c = [0.0, 0.0]
        self.d_b = [0.0, 0.0, 0.0]

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + m.exp(-x))

    @staticmethod
    def softmax(x):
        exp_x = [m.exp(xi) for xi in x]
        sum_exp_x = sum(exp_x)
        return [xi / sum_exp_x for xi in exp_x]

    def forward(self):
        for j in range(len(self.k)):
            for i in range(len(self.x)):
                self.k[j] += self.w[i][j] * self.x[i]
            self.k[j] += self.bias_x[j]

        for i in range(len(self.k)):
            self.h[i] = self.sigmoid(self.k[i])

        for i in range(len(self.o)):
            for j in range(len(self.h)):
                self.o[i] += self.v[j][i] * self.h[j]
            self.o[i] += self.bias_c[i]

        self.y = self.softmax(self.o)

        # TODO: Check if this is correct.
        self.L = sum(-m.log(y) * t for y, t in zip(self.y, self.t))

    def backward(self):
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
