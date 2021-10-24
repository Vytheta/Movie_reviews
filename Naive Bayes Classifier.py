import numpy as np


def train_nb(X, Y):

    """Train a binary NB classifier."""
    # + 1 for the Laplacian smoothing
    pos_p = X[Y == 1, :].sum(0) + 1
    pos_p = pos_p / pos_p.sum()
    neg_p = X[Y == 0, :].sum(0) + 1
    neg_p = neg_p / neg_p.sum()
    w = np.log(pos_p) - np.log(neg_p)
    # Estimate P(0) and P(1) and compute b
    b = 0
    return w, b


def nb_inference(X, w, b):
    """Prediction of a binary NB classifier."""
    logits = X @ w + b
    labels = (logits > 0).astype(int)
    return labels, logits        # No need to change


# The script loads the training data and train a classifier.  It must
# be extended to evaluate the classifier on the test set.
data = np.loadtxt("train.txt.gz")
X = data[:, :-1]
Y = data[:, -1]
w, b = train_nb(X, Y)
predictions = nb_inference(X, w, b)[0]
accuracy = (predictions == Y).mean()
print("Training accuracy:", accuracy * 100)



data = np.loadtxt("test.txt.gz")
X = data[:, :-1]
Y = data[:, -1]
predictions = nb_inference(X, w, b)[0]

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
predictions = nb_inference(X, w, b)[0]
accuracy = (predictions == Y).mean()
print("Validating accuracy:", accuracy * 100)

print('Test overestimation vocabulary word n°', np.argmax(T) + 1)
print('Test underestimation vocabulary word n°', np.argmin(T) + 1)




# This part detects the most relevant words for the classifier.
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
