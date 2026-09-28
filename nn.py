"""Нейросеть с нуля на numpy."""
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layers):
        """layers = [input_size, hidden_size, output_size]"""
        self.W1 = np.random.randn(layers[0], layers[1]) * 0.01
        self.b1 = np.zeros((1, layers[1]))
        self.W2 = np.random.randn(layers[1], layers[2]) * 0.01
        self.b2 = np.zeros((1, layers[2]))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y, lr=0.01):
        m = X.shape[0]
        dz2 = self.a2 - y
        dW2 = self.a1.T @ dz2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_deriv(self.z1)
        dW1 = X.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def loss(self, y_pred, y_true):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

# === Генерация данных ===
np.random.seed(42)
n = 1000
X = np.random.randn(n, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)
y_onehot = np.eye(2)[y]

# === Обучение ===
nn = NeuralNetwork([2, 16, 2])
losses = []
for epoch in range(500):
    y_pred = nn.forward(X)
    loss = nn.loss(y_pred, y_onehot)
    losses.append(loss)
    nn.backward(X, y_onehot, lr=0.1)
    if epoch % 50 == 0:
        acc = np.mean(np.argmax(y_pred, axis=1) == y)
        print(f"Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}")

# === Визуализация ===
plt.plot(losses)
plt.title("Loss curve")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.savefig("loss.png")
plt.show()
