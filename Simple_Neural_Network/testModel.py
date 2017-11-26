import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()

mnist = input_data.read_data_sets("dataset/", one_hot=True, reshape=False)

n_inputs = 784
n_hidden = 256
n_classes = 10

# Features and Labels
x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_classes])
x_flat = tf.reshape(x, [-1, n_inputs])

# Weights and Biases
biases = {
			'hidden': tf.Variable(tf.random_normal([n_hidden]), name='bias_hidden'),
			'output': tf.Variable(tf.random_normal([n_classes]), name='bias_output')
}

weights = {
			'hidden': tf.Variable(tf.random_normal([n_inputs, n_hidden]), name='weights_hidden'),
			'output': tf.Variable(tf.random_normal([n_hidden, n_classes]), name='weights_output')
}


keep_prob = tf.placeholder(tf.float32)

# Layers of DNN
hidden_layer = tf.add(tf.matmul(x_flat, weights['hidden']), biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)

logits = tf.add(tf.matmul(hidden_layer, weights['output']), biases['output'])

# Accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Saver
saver = tf.train.Saver()
save_file = 'checkpoint/trained_model.ckpt'

with tf.Session() as sess:
	# Restore trained weights
	saver.restore(sess, save_file)

	test_accuracy = sess.run(accuracy, feed_dict = {x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})

	print("Test Accuracy = ", test_accuracy)
