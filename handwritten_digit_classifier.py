if __name__ == '__main__':
    import NN
    import mnist_loader as ml

    training_data,validation_data,test_data=ml.load_data_wrapper();
    """
    Initializing a deep neural network with 30 neurons in hidden layer.
    Custom NN architecture is also welcome.
    """
    net=NN.NN([784,30,10]);

    """
    Training goes on for a default of 10 epochs. The accuracy over both the training and the validation datasets is printed after each epoch.
    """
    net.train(training_data,learning_rate=3.0,mini_batch_size=50,validation_data=validation_data)

    print("Accuracy over test data = {0}".format(net.test(test_data)))

    """
    # Code snippet to store the trained weights and biases

    np.save("digit_classifier_weights",net.w)
    np.save("digit_classifier_biases",net.b)
    """
