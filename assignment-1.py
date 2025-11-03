# %% Cell 1
class simple_NN:
    def __init__(self, input_x: list, target: list) -> None:
        self.x = input_x
        self.t = target
        self.k = [0.0, 0.0, 0.0]
        self.h = [0.0, 0.0, 0.0]
        self.w = [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]
        self.v = [[1.0, 1.0], [-1.0, -1.0], [-1, -1]]
        self.bias_x = [0.0, 0.0, 0.0]
        self.bias_c = [0.0, 0.0]
        self.o = [0.0, 0.0]
        self.L = 0.0

        # Derivatives
        self.d_x = []
        self.d_o = []
        self.d_v = []
        self.d_h = []
        self.d_w = []
        self.d_k = []

    def forward(self):
        for j in range(len(self.k)):
            for i in range(len(self.x)):
                self.k[j] += self.w[i][j] * self.x[i] + self.bias_x[j]

    def backward(self):
        pass
