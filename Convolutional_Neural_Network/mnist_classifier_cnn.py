from tf_CNN import tf_CNN as CNN
from tf_CNN_layers import ConvPoolLayer, ConnectedLayer, SoftmaxOutputLayer
import mnist_loader as ml

tr_d, va_d, te_d = ml.load_data_wrapper()

cnet = CNN(
            [
                ConvPoolLayer(
                                (50,28,28,1),
                                (5,5,20),
                                1,
                                (2,2),
                            ),
                ConvPoolLayer(
                                (50,12,12,20),
                                (3,3,16),
                                1,
                                (2,2),
                                pool_stride=2,
                                linear_output=True,
                            ),
                ConnectedLayer(
                                n_in=5*5*16,
                                n_out=1000,
                                mini_batch_size=50,
                            ),
                SoftmaxOutputLayer(
                                n_in=1000,
                                n_out=10,
                                mini_batch_size=50,
                            )
            ]
            )
cnet.train(tr_d)
