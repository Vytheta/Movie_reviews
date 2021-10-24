import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logreg_inference(X, w, b):
    logits = (X @ w) + b
    P = sigmoid(logits)
    return P

def sum(Y, p):
    d = 0
    for i in range(25000):
        d += abs(Y[i] - p[i])
    return d

def sum2(Y, p):
    d = 0
    for i in range(12500):
        d += abs(Y[i] - p[i])
    return d


def logreg_train(X, Y):
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    lr = 0.3
    for step in range(5000):
        P = logreg_inference(X, w, b)
        grad_w = (X.T @ (P - Y)) / m
        grad_b = (P - Y).mean()
        w = w - lr * grad_w
        b = b - lr * grad_b
        p = logreg_inference(X, w, b)
        n = 1 - sum(Y, p) / 25000
        #print("training accuracy is", n * 100, "%")
    return w, b


data = np.loadtxt("train.txt.gz")
X = data[:, :1000]
Y = data[:, 1000]
w, b = logreg_train(X, Y)
p = logreg_inference(X, w, b)
n = 1 - sum(Y, p) / 25000
print("training accuracy is", n * 100, "%")

data = np.loadtxt("test.txt.gz")
X = data[:, :1000]
Y = data[:, 1000]
p = logreg_inference(X, w, b)
n = 1 - sum2(Y, p) / 12500
print("testing accuracy is", n * 100, "%")

data = np.loadtxt("validation.txt.gz")
X = data[:, :1000]
Y = data[:, 1000]
p = logreg_inference(X, w, b)
n = 1 - sum2(Y, p) / 12500
print("validation accuracy is", n * 100, "%")


f = open("vocabulary1")
voc = f.read().split()
f.close()

indices = w.argsort()
print("NEGATIVE WORDS")
for i in indices[:20]:
    print(voc[i], w[i])

print()
print("POSITIVE WORDS")
for i in indices[-20:]:
    print(voc[i], w[i])

