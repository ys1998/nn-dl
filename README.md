# Neural Networks and Deep Learning
This repository is a collection of resources that I used to dive into the field of neural networks and deep learning, and implementations of different kinds of neural networks in Python 3:
* from scratch
* using TensorFlow (without using inbuilt classes)

## Code

* Feedforward neural network \[ [scratch](NN.py) | [TensorFlow](tf_DNN.py) ]
* vanilla Recurrent Neural Network \[ [scratch](Recurrent_Neural_Networks/RNN.py) | [TensorFlow](Recurrent_Neural_Networks/tf_RNN.py) ]
* MNIST handwritten digit classification \[ [scratch](handwritten_digit_classifier.py) | [TensorFlow](tf_handwritten_digit_classifier.py) ]
* Basic language modeling on PTB corpus using vanilla RNN \[ [scratch](Recurrent_Neural_Networks/basic_language_modeling.py) | [TensorFlow](Recurrent_Neural_Networks/tf_basic_language_modeling.py) ]

## Datasets and helper functions
* MNIST \[ [dataset](data/MNIST/) | [helper code](mnist_loader.py) ]
* PTB \[ [dataset](data/PTB/) | [helper code](Recurrent_Neural_Networks/ptb_loader.py) ]

## Resources
Most of the code is inspired from the online book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com) by M. Nielson (pdf version of the same along with a chapter-wise summary can be found [here](Book/)). Apart from this, the following resources were referred to for better understanding -

1. Colah's Blog - **Understanding LSTM Networks** \[ [pdf](Reference/Understanding_LSTM_Networks_colah's_blog.pdf) | [html](https://http://colah.github.io/posts/2015-08-Understanding-LSTMs/) ]
2. Andrej Karpathy's Blog - **The Unreasonable Effectiveness of Recurrent Neural Networks** \[ [pdf](Reference/The_Unreasonable_Effectiveness_of_Recurrent_Neural_Networks.pdf) | [html](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) ]
3. WildML - **Introduction to Recurrent Neural Networks** \[ [pdf](Reference/WildML_Intro_to_RNNs.pdf) | [html](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) ]
4. WildML - **Understanding BPTT** \[ [pdf](Reference/WildML_Understanding_BPTT.pdf) | [html](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/) ]
5. **Gradients for an RNN** - Carter N. Brown \[ [pdf](Reference/BPTT_proof.pdf) ]
6. **Vector, Matrix, Tensor Derivatives** - Eric Miller \[ [pdf](Reference/Derivatives_of_tensors.pdf) ]
7. **Deep Learning** - Ian Goodfellow, Yoshua Bengio, Aaron Courville \[ [pdf](Reference/Deep_Learning.pdf) | [html](http://www.deeplearningbook.org/) ]
8. **Tutorial on DNNs** - Google Codelabs \[ [html](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0) ]
9. **Different architectures for RNNs** - Wikipedia \[ [html](https://en.wikipedia.org/wiki/Recurrent_neural_network#Architechtures) ]

