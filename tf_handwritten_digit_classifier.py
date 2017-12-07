import tf_DNN
import mnist_loader as ml

tr_d, va_d, te_d = ml.load_data_wrapper()
# Custom structures are possible - just change the layer sizes/number
net=tf_DNN.DNN([784,30,10])
net.train(tr_d)
