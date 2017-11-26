
# Logistic Classifier for Hand-Written Digits Recognition

This program classifies images of hand-written digits in MNIST dataset using logistic classifier

### Read and pre-process data


```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Read mnist dataset
mnist = input_data.read_data_sets("../Datasets/MNIST/", one_hot=True)
```

    Extracting ../Datasets/MNIST/train-images-idx3-ubyte.gz
    Extracting ../Datasets/MNIST/train-labels-idx1-ubyte.gz
    Extracting ../Datasets/MNIST/t10k-images-idx3-ubyte.gz
    Extracting ../Datasets/MNIST/t10k-labels-idx1-ubyte.gz


### Configure


```python
n_features = 784
n_labels = 10
n_epochs = 100
batch_size = 256
learn_rate = 0.003

features = tf.placeholder("float", shape=([None, n_features]), name = 'features')
labels = tf.placeholder("float", shape=([None, n_labels]), name = 'labels')
weights = tf.Variable(tf.random_normal(shape=([n_features, n_labels])), name = 'weights')
biases = tf.Variable(tf.zeros(n_labels), name = 'biases')
```

### Logistic Classifier
A logistic classifier is a linear classifier, it takes inputs, which are pixels in this problem, and applies linear function (xW + b) to them to generate predictions. The output of linear function are called logits, and they represent the score for each class in the dataset. The scores are converted to probability using Softmax function. Further, the labels in the dataset are one-hot encoded. In order to compare the probabilites with the one-hot encoded labels, we use cross-entropy. A cross-entropy, in layman terms, measures distance between the probabilies and the one hot-encoded values. Once we find the distance between predicted value and the actual value, we calculate the loss as mean of cross-entropy over entire dataset. The loss is minimized and correct weights and bias are learned using Gradient-Descent optimizer.

### Why am I using Softmax function?
Softmax function is a non-linear function and it converts a linear input to non-linear output. This is important in multi-layer neural network because if we don't use a non-linear activation function, then no matter how deep our neural network is, it can basically be represented by a single layer neural network (because linear combination of linear combinations is again a linear combination), and we gain nothing from adding multiple layers. So in order to take benefit of multiple layers in a neural network, we use non-linear activation functions like softmax, ReLU, tanh, sigmoid etc... And most of the real data available is not linearly separable, we need a non-linear model to model such data.


```python
logits = tf.add(tf.matmul(features, weights), biases)

# Below 3 lines can be written with a single command: tf.nn.softmax_cross_entropy_with_logits()
softmax = tf.nn.softmax(logits = logits)
cross_entropy = -tf.reduce_sum(tf.multiply(labels, tf.log(softmax)))
loss = tf.reduce_mean(cross_entropy)

# Using Gradient Descent to learn correct weights and biases
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learn_rate).minimize(loss)

# Calculate prediction accuracy
correct_predictions = tf.equal(tf.argmax(logits, axis = 1), tf.argmax(labels, axis = 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
```

#### Train model


```python
with tf.Session() as sess:
    # Initialize tf Variables
    sess.run(tf.global_variables_initializer())
    
    # Train over n_epochs
    for e in range(n_epochs):
        # Divide dataset into batches and train the model over the batches for each epoch
        num_batches = int(mnist.train.num_examples/batch_size)
        for i in range(num_batches):
            features_batch, labels_batch = mnist.train.next_batch(batch_size)
            sess.run(optimizer, 
                     feed_dict = {
                                     features : features_batch, 
                                     labels : labels_batch
                                 })
        
        # Print learning progress for every 10 epochs
        if(e%10 == 0):
            output = sess.run(accuracy, 
                              feed_dict={
                                          features : mnist.validation.images, 
                                          labels : mnist.validation.labels
                                        })
            print("Epoch = ", e+10, ", Validation Accuracy = ", output)
```

    Epoch =  10 , Validation Accuracy =  0.8448
    Epoch =  20 , Validation Accuracy =  0.9034
    Epoch =  30 , Validation Accuracy =  0.9112
    Epoch =  40 , Validation Accuracy =  0.9164
    Epoch =  50 , Validation Accuracy =  0.9184
    Epoch =  60 , Validation Accuracy =  0.9168
    Epoch =  70 , Validation Accuracy =  0.9216
    Epoch =  80 , Validation Accuracy =  0.924
    Epoch =  90 , Validation Accuracy =  0.9262
    Epoch =  100 , Validation Accuracy =  0.9228

