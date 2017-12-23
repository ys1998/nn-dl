import sys
import tf_RNN
import tf_LSTM
import ptb_loader as pl
import tensorflow as tf

print("Procuring training data ...")
I,O,index_to_word = pl.get_data_and_dict(data_size=-1,batch_size=50,bptt_steps=10)
print("Data obtained.")

# print(sys.argv[1])
if sys.argv[1]=='1':
    print("Constructing RNN ...")
    rnet=tf_RNN.tf_RNN(len(index_to_word),50,50,10,activation=tf.nn.sigmoid)
    # Train the RNN
    print("Training RNN ...")
    rnet.train(I,O,learning_rate=1.0,n_epochs=30)
    # Predict sentences
    # print("Predicting sentence ...")
    # rnet.predict(index_to_word)
elif sys.argv[1]=='2':
    print("Constructing LSTM ...")
    lstm=tf_LSTM.tf_LSTM(len(index_to_word),50,10)
    print("Training LSTM ...")
    lstm.train(I,O,learning_rate=1.0,n_epochs=30)
