import numpy as np


def relu(x):
    """ReLU激活函数"""
    return np.maximum(0, x)


def relu_derivative(x):
    """ReLU激活函数的导数"""
    return np.where(x > 0, 1, 0)


class SimpleNeuralNetwork:
    def __init__(self, input_dim, output_dim):
        """初始化网络参数"""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.randn(output_dim, input_dim) * 0.01
        self.biases = np.zeros((output_dim, 1))

    def forward(self, x):
        """前向传播"""
        self.z = np.dot(self.weights, x) + self.biases
        self.a = relu(self.z)
        return self.a

    def compute_loss(self, predicted, actual):
        """计算损失"""
        return np.mean((predicted - actual) ** 2)

    def backward(self, x, y):
        """反向传播"""
        # 计算损失相对于激活的导数
        dA = 2 * (self.a - y) / y.size
        # 计算激活函数相对于z的导数
        dZ = dA * relu_derivative(self.z)
        # 计算z相对于权重的导数
        dW = np.dot(dZ, x.T)
        # 计算z相对于偏置的导数
        dB = np.sum(dZ, axis=1, keepdims=True)
        return dW, dB

    def update_params(self, dW, dB, lr):
        """更新权重和偏置"""
        self.weights -= lr * dW
        self.biases -= lr * dB

    def train(self, x, y, lr, n_iters):
        """训练模型"""
        for i in range(n_iters):
            output = self.forward(x)
            loss = self.compute_loss(output, y)
            dW, dB = self.backward(x, y)
            self.update_params(dW, dB, lr)
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss}")


# 示例数据
np.random.seed(0)
input_vector = np.random.randn(3, 1)  # 输入数据 (input_dim, 1)
actual_output = np.array([[1], [0]])  # 实际输出 (output_dim, 1)

# 创建和训练网络
network = SimpleNeuralNetwork(input_dim=3, output_dim=2)
network.train(input_vector, actual_output, lr=0.01, n_iters=1000)