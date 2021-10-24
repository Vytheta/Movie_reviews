import collections
import os
from nltk.stem import SnowballStemmer
import spacy_lookups_data as lsd

h = open("newlist", "r")             # newlist = stopwords list as a sentence
string = h.read()
l = string.split()

def not_common_word(word):
    """Define if a word is common or uncommon in respect to the stopwords.txt file"""
    n = 0
    for j in range(319):
        if word.lower() == l[j]:
            n += 1
    #print(word, n, n==0)
    if n == 0:
        return True
    else:
        return False


def read_document(filename):
    """Read the file and returns a list of words."""
    f = open(filename, encoding="utf8")
    text = f.read()
    f.close()
    # The three following lines replace punctuation symbols with
    # spaces.
    p = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~0123456789"
    table = str.maketrans(p, " " * len(p))
    text = text.translate(table)
    words = []
    for w in text.split():
        w = w.lower()
        #w = w.lemma_
        #sbs = SnowballStemmer(language='english')
        #w = sbs.stem(w)
        if (len(w) > 2 and not_common_word(w)):
            words.append(w)
            #print(w)
    #print(words)
    return words


def write_vocabulary(voc, filename, n):
    """Write the n most frequent words to a file."""
    f = open(filename, "w")
    for word, count in sorted(voc.most_common(n)):
        print(word, file=f)
    f.close()


# The script reads all the documents in the smalltrain directory, uses
# the to form a vocabulary, writes it to the 'vocabulary_train.txt' file.

voc = collections.Counter()

for f in os.listdir("aclImdb/train/pos"):
    voc.update(read_document("aclImdb/train/pos/" + f))
#print(voc)
for f in os.listdir("aclImdb/train/neg"):
    voc.update(read_document("aclImdb/train/neg/" + f))
    #print(voc)
write_vocabulary(voc, "vocabulary1", 1000)

