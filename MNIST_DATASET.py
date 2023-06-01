import numpy as np
import requests, gzip
import matplotlib.pyplot as plt

from pathlib import Path

#----------------------------------------------------------------------------------------------
# taken from https://ludius0.github.io/my-blog/ai/deep%20learning%20(dl)/2020/12/14/Neural-network-from-scratch.html

def fetch(url):    
    name = url.split("/")[-1]
    dirs = Path("dataset/mnist")
    path = (dirs / name)
    if path.exists():
        with path.open("rb") as f:
            dat = f.read()
    else:
        if not dirs.is_dir(): dirs.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            dat = requests.get(url).content
            f.write(dat)
    return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()

def one_hot_encoding(y):
    N = y.shape[0]
    Z = np.zeros((N, 10), np.float32)
    Z[np.arange(N), y] = -1.0 # correct loss for NLL, torch NLL loss returns one per row
    return Z

def normalize(x): 
    return (x - np.mean(x, dtype=np.float32)) / np.std(x, dtype=np.float32)

def fetch_mnist():
    print(" collecting data")
    X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

    ## pre processing data
    X_train = normalize(X_train) 
    Y_train = one_hot_encoding(Y_train)

    #X_test = normalize(X_test)
    #Y_test = one_hot_encoding(Y_test)
    return (X_train, Y_train, X_test, Y_test)

if __name__ == "__main__":
  print("ERROR")
  X_train, Y_train, X_test, Y_test = fetch_mnist()
  for i in range(3):
    img = X_test[i].reshape((28,28))
    plt.imshow(img, cmap="Greys")
    plt.title( str(np.argmax(Y_test[i])) )
    plt.show()

