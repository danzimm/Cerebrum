#!/usr/bin/env bash

downloadIdx() {
  echo "curl $1.gz | gzip -d > $(basename $1).idx"
  curl $1.gz | gzip -d > $(basename $1).idx
}

downloadIdx http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte
downloadIdx http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte
downloadIdx http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte
downloadIdx http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte

