# Simple Neural Network

This program implements a simple 3 layer (input, hidden, output) layer neural network with softmax regression. <br />
The program trainModel.py uses TensorFlow saver to save the trained model, and testModel.py uses the saved model to test prediction accuracy on test set. <br />

## Sample output of trainModel.py
Extracting dataset/train-images-idx3-ubyte.gz <br />
Extracting dataset/train-labels-idx1-ubyte.gz <br />
Extracting dataset/t10k-images-idx3-ubyte.gz <br />
Extracting dataset/t10k-labels-idx1-ubyte.gz <br />
Epoch =  0 Validation_Accuracy =  0.476 <br />
Epoch =  10 Validation_Accuracy =  0.8338 <br />
Epoch =  20 Validation_Accuracy =  0.8678 <br />
Epoch =  30 Validation_Accuracy =  0.8802 <br />
Epoch =  40 Validation_Accuracy =  0.8868 <br />
Saved trained model <br />

## Sample output of testModel.py
Extracting dataset/train-images-idx3-ubyte.gz <br />
Extracting dataset/train-labels-idx1-ubyte.gz <br />
Extracting dataset/t10k-images-idx3-ubyte.gz <br />
Extracting dataset/t10k-labels-idx1-ubyte.gz <br />
Test Accuracy =  0.8906 <br />
