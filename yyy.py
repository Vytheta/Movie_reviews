import os


for f in os.listdir("aclImdb/train/pos/"):
    g = open(f, encoding="utf8")
    text = g.read()
    g.close()
    print(text)