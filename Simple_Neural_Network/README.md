# Simple Neural Network

This program implements a simple 3 layer (input, hidden, output) layer neural network with softmax regression.
The program trainModel.py uses TensorFlow saver to save the trained model, and testModel.py uses the saved model to test prediction accuracy on test set.

## Sample output of trainModel.py
Extracting dataset/train-images-idx3-ubyte.gz
Extracting dataset/train-labels-idx1-ubyte.gz
Extracting dataset/t10k-images-idx3-ubyte.gz
Extracting dataset/t10k-labels-idx1-ubyte.gz
Epoch =  0 Validation_Accuracy =  0.476
Epoch =  10 Validation_Accuracy =  0.8338
Epoch =  20 Validation_Accuracy =  0.8678
Epoch =  30 Validation_Accuracy =  0.8802
Epoch =  40 Validation_Accuracy =  0.8868
Saved trained model


## Sample output of testModel.py
