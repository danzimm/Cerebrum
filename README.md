Cerebrum
---
This is my pet project to learn the basics of neural nets. I'm following along the book [over here](http://neuralnetworksanddeeplearning.com). I find the book easy to read, and I was recommended it by [@mcnees](https://twitter.com/mcnees).

Getting Started
---
First run `./download.sh` to download the training and test data from the [MNIST database](http://yann.lecun.com/exdb/mnist/). Now `make` to make what is available. See below for more information about what is available.

`load_labels`
---
This is to examine labels files. Run `./load_labels [some_labels_file]` to view an output of the expected labels for that set of data.

`load_images`
---
This is to examine images files. Run `./load_images [some_images_file]` to view the images embedded in that file (I use ascii art to display right in your terminal).

Roadmap
---
[x] Create IDX file reader
[x] Create IDX file reader tests
  [x] Read in labels files
  [x] Read in images files
[ ] Create Neuron (a general form of a Sigmoid Neuron where the SN is just a specialization of the Neuron)
[ ] Create SN tests
  [ ] Train and
  [ ] Train or
  [ ] Train sum with carry
  [ ] Train digit recognizer
[ ] Create a convolutional neural network
[ ] Create CNN tests
  [ ] Train to recognize an icon
[ ] Create recurrent neural network
[ ] Implement [this](http://mi.eng.cam.ac.uk/~thw28/papers/TR698.pdf) to generate wookieleaks

Contributing
---
This really isn't in any sort of stage where contribution will be accepted - it's mostly a personal learning project that's being open sourced in case anybody wants to see an example of some working (hopefully) code.
