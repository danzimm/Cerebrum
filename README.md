Cerebrum
---
This is my pet project to learn the basics of neural nets. I'm following along the book [over here](http://neuralnetworksanddeeplearning.com). I find the book easy to read, and I was recommended it by [@mcnees](https://twitter.com/mcnees).

Status
---
![travis-ci](https://travis-ci.org/danzimm/Cerebrum.svg?branch=master)

Getting Started
---
First run `./download.sh` to download the training and test data from the [MNIST database](http://yann.lecun.com/exdb/mnist/). Now `make` to make what is available. See below for more information about what is available. Whenever contributing code, be sure to add tests. To run the tests do `make test`.

`load_labels`
---
This is to examine labels files. Run `./load_labels [some_labels_file]` to view an output of the expected labels for that set of data.

`load_images`
---
This is to examine images files. Run `./load_images [some_images_file]` to view the images embedded in that file (I use ascii art to display right in your terminal).

`tests`
---
This just runs the tests that are in the project. Be sure include your test implementation files to the test target in the makefile. To declare a test use the macro `DeclareTest(clsName, testName)`. See [MatrixTests.cc](MatrixTests.cc) for an exmaple. You can also create your own test suite if you're adding an entire module worth of features.

Roadmap
---
- [ ] Basic CNN
  - [ ] Figure out features
  - [ ] Figure out tests
- [ ] RNN
  - [ ] Implement [this](http://mi.eng.cam.ac.uk/~thw28/papers/TR698.pdf) to generate wookieleaks
  - [ ] Figure out features
  - [ ] Figure out tests
- [ ] [Hopfiled Network](http://www.wikiwand.com/en/Hopfield_network)

Contributing
---
If you want to contribute feel free. This is an attempt to get a c++ ML library underway - just for fun. If you contribute be sure to write tests for the code you create. I'd like to stay on track with the roadmap, but other ML utilities are also appreciated! Please follow the code style that has been put in place - there are no explicit rules, just try to follow the style of the code already written. I can say 2 spaces for tabs. Both vim and emacs is welcome.
