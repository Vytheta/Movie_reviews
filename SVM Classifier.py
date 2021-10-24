import numpy as np

def svm_inference(X, w, b):
    logits = X @ w + b
    labels = (logits > 0).astype(int)
    return labels, logits

def hinge_loss(labels, logits):
    loss = np.maximum(0, 1 - (2 * labels - 1) * logits)
    return loss.mean()

def svm_train(X, Y, lambda_, lr=0.1, steps=10000, init_w=None, init_b=0):
    m, n = X.shape
    w = (init_w if init_w is not None else np.zeros(n))
    b = init_b
    C = (2 * Y) - 1
    for step in range(steps):
        labels, logits = svm_inference(X, w, b)
        hinge_diff = -C * ((C * logits) < 1)
        grad_w = (hinge_diff @ X) / m + lambda_ * w
        grad_b = hinge_diff.mean()
        w -= lr * grad_w
        b -= lr * grad_b
        #print(hinge_loss(labels, logits))
    return w, b


data = np.loadtxt("train.txt.gz")
X = data[:, :-1]
Y = data[:, -1]
w, b = svm_train(X, Y, 1e-5)
predictions = svm_inference(X, w, b)[0]
accuracy = (predictions == Y).mean()
print("Training accuracy:", accuracy * 100)



data = np.loadtxt("test.txt.gz")
X = data[:, :-1]
Y = data[:, -1]
predictions = svm_inference(X, w, b)[0]

T = np.zeros(1000)
for i in range(12500):
    if predictions[i] > Y[i]:
        for j in range(1000):
            if w[j] > 0:
                T[j] += w[j]*X[i][j]
    else:
        if predictions[i] < Y[i]:
            for k in range(1000):
                if w[k] < 0:
                    T[k] += w[k]*X[i][k]

accuracy = (predictions == Y).mean()
print("Testing accuracy:", accuracy * 100)



data = np.loadtxt("validation.txt.gz")
X = data[:, :-1]
Y = data[:, -1]
predictions = svm_inference(X, w, b)[0]
accuracy = (predictions == Y).mean()
print("Validating accuracy:", accuracy * 100)

print('overestimation vocabulary word n°', np.argmax(T) + 1)
print('underestimation vocabulary word n°', np.argmin(T) + 1)

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
