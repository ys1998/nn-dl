# Keywords summary

## Chapter 1
- perceptrons (binary output - 0/1)
- sigmoid neurons
- weights, biases/thresholds
- input/output/hidden layer(s)
- error/cost/loss/performance function
- gradient descent
- stochastic gradient descent
- mini-batch
- epoch
- online/on-line/incremental learning (mini-batch size 1)
- deep neural networks

## Chapter 2

- backpropagation algorithm
- weight, bias, activation input and weighted input (matrix notation for each)
- Hadamard/Schur product (elementwise multiplication)
- four equations of backpropagation

## Chapter 3

- binary entropy function (for bernoulli distributions)
- cross entropy function
- backpropagation algorithm for above (changes only in the last layer deltas)
- learning slowdown / saturation
- softmax layer of outputs
  - **Fun fact :** origin of the term "softmax" (softened-maximum)
- log-likelihood loss function
- overfitting

### Methods to prevent overfitting
 - early stopping (using validation data)
 - hold out method (again using "separate" validation data)
 - regularization

### Methods of regularization
 - weight decay or L2 regularization
   - formula (addition of square of weights to old cost function)
   - regularization parameter `lambda`
   - changes in backpropagation formula
 - L1 regularization
   - formula (addition of absolute value of weights to original cost function)
   - changes in backpropagation algorithm
 - Dropout
   - `pkeep` parameter : fraction of neurons NOT discarded
   - scale the parameters accordingly to account for them being counted multiple times
 - Artificial generation of test data
   - images - rotation by small amounts, slight distortions
   - sound - alter speed, background noise, pitch

### Methods for faster learning
- weight initialization
  - xavier initialization ( initialization with `N(0,1/sqrt(n_neurons))` )

- Hyperparameter tuning *(last two methods aren't mentioned in the book)*
  - empirical testing (naive)
  - grid method vs random method (paper by Yoshua Bengio)
  - Bayesian method

- Alternate optimization functions
  - Gradient descent with decaying learning rate
  - Hessian-based descent
    - delta_w = - (inverse of Hessian matrix)*(gradient of cost function)
    - computationally expensive
  - momentum-based gradient descent
    - formula (maintain a *velocity* vector for each weight vector)
    - momentum coefficient, mu
    - easy to compute, faster
  - *Extra (TensorFlow) :* Adam optimizer
    - implements momentum-based GD and decaying learning rate ??

## Chapter 4

 - Universality Theorem for neural networks

## Chapter 5

 - instabiliy of gradient in earlier layers
 - vanishing gradient problem
 - exploding gradient problem
 - effects of random initialization and activation functions
 - effects of optimizers (e.g. how gradient descent is applied, etc.)

## Chapter 6

 - convolutional neural networks
   - based on local receptive fields, shared weights, pooling
   - adapted to translational invariance of images
   - standard architechture of CNNs

### More on CNN's
 - local receptive fields
   - hidden layer(s) and connections between neurons
   - stride length
 - shared weights and biases
   - feature and feature maps (map input layer to hidden layer)
 - pooling
   - pooling layer generally used after a convolutional layer
   - input regions may or may not overlap (generally they are disjoint)
   - types -> max-pooling, L2 pooling
   - the shared weights/bias for each layer define a kernel/filter
 - common architechture: input, convolutional, max-pool, fully-connected hidden layer, output layer with softmax activation and log-likelihood loss function
 - multi-convolutional-layer networks
 - regularization techniques - dropout (for fully connected layers), no regularizers for convolutional layers due to inbuilt *resistance*
 - ensemble of networks 

### Recent topics
 - adversarial images/input
 - recurrent neural networks
   - interesting application : Neural Turing Machines (NTMs)
 - Long Short-Term Memory networks (LSTMs)
 - deep belief networks, generative models and Boltzmann machines
 - deep learning and AI
