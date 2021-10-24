import numpy as np
import os
from nltk.stem import PorterStemmer


def load_vocabulary(filename):
    f = open(filename)
    n = 0
    voc = {}
    for w in f.read().split():
        voc[w] = n
        n += 1
    f.close()
    return voc


def read_document(filename, voc):
    f = open(filename, encoding="utf8")
    text = f.read()
    f.close()
    p = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    table = str.maketrans(p, " " * len(p))
    text = text.translate(table)
    # Start with all zeros
    bow = np.zeros(len(voc))
    #ps = PorterStemmer()
    for w in text.split():
        # If the word is the vocabulary...
        #w = ps.stem(w.lower())
        if w in voc:
            # ...increment the proper counter.
            index = voc[w]
            bow[index] += 1
    return bow


# The script compute the BoW representation of all the training
# documents.  This need to be extended to compute similar
# representations for the validation and the test set.
voc = load_vocabulary("vocabulary1")
documents = []
labels = []
for f in os.listdir("aclImdb/train/pos"):
    documents.append(read_document("aclImdb/train/pos/" + f, voc))
    labels.append(1)
for f in os.listdir("aclImdb/train/neg"):
    documents.append(read_document("aclImdb/train/neg/" + f, voc))
    labels.append(0)
# np.stack transforms the list of vectors into a 2D array.
X = np.stack(documents)
Y = np.array(labels)
# The following line append the labels Y as additional column of the
# array of features so that it can be passed to np.savetxt.
data = np.concatenate([X, Y[:, None]], 1)
print(data)
np.savetxt("train.txt.gz", data)

documents_test = []
labels_test = []
for f in os.listdir("aclImdb/test/pos"):
    documents_test.append(read_document("aclImdb/test/pos/" + f, voc))
    labels_test.append(1)
for f in os.listdir("aclImdb/test/neg"):
    documents_test.append(read_document("aclImdb/test/neg/" + f, voc))
    labels_test.append(0)
# np.stack transforms the list of vectors into a 2D array.
X = np.stack(documents_test)
Y = np.array(labels_test)
# The following line append the labels Y as additional column of the
# array of features so that it can be passed to np.savetxt.
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("test.txt.gz", data)

documents_validation = []
labels_validation = []
for f in os.listdir("aclImdb/validation/pos"):
    documents_validation.append(read_document("aclImdb/validation/pos/" + f, voc))
    labels_validation.append(1)
for f in os.listdir("aclImdb/validation/neg"):
    documents_validation.append(read_document("aclImdb/validation/neg/" + f, voc))
    labels_validation.append(0)
# np.stack transforms the list of vectors into a 2D array.
X = np.stack(documents_validation)
Y = np.array(labels_validation)
# The following line append the labels Y as additional column of the
# array of features so that it can be passed to np.savetxt.
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("validation.txt.gz", data)



