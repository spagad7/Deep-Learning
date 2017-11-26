import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()

# Read dataset
mnist = input_data.read_data_sets("../Datasets/MNIST/", one_hot=True, reshape=False)

# Set parameters
n_inputs = 784
n_hidden = 256
n_classes = 10
n_epochs = 50
learn_rate = 0.001
batch_size = 128

# Init weights and biases
weights = {
			'hidden': tf.Variable(tf.random_normal([n_inputs, n_hidden]), name='weights_hidden'),
		   	'output': tf.Variable(tf.random_normal([n_hidden, n_classes]), name='weights_output')
		  }

biases = {
			'hidden': tf.Variable(tf.random_normal([n_hidden]), name='bias_hidden'),
			'output':tf.Variable(tf.random_normal([n_classes]), name='bias_output')
		 }

# Define inputs and outputs
x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_classes])
x_flat = tf.reshape(x, [-1, n_inputs])

keep_prob = tf.placeholder(tf.float32)

# Define hidden and output layers
logits_hidden_layer = tf.add(tf.matmul(x_flat, weights['hidden']), biases['hidden'])
activation_hidden_layer = tf.nn.relu(logits_hidden_layer)
dropout_hidden_layer = tf.nn.dropout(activation_hidden_layer, keep_prob)

# Output_layer
logits_output_layer = tf.add(tf.matmul(dropout_hidden_layer, weights['output']), biases['output'])

# Define cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits_output_layer, labels=y))

# Define optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learn_rate).minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits_output_layer, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Saver
saver = tf.train.Saver()
save_file = 'checkpoint/trained_model.ckpt'

init = tf.global_variables_initializer()

# Run
with tf.Session() as sess:
	# Init global
	sess.run(init)

	# Run for defined number of epochs
	for epoch in range(n_epochs):
		# Get total number of batches to iterate
		total_batch_num = int(mnist.train.num_examples / batch_size)

		# Iterate through each batch
		for i in range(total_batch_num):
			# Get batch training features and labels
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			# Run optimizer
			sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

		# Print accuracy for every 10 epochs
		if epoch % 10 == 0:
			validation_accuracy = sess.run(accuracy, feed_dict = {x: mnist.validation.images, y: mnist.validation.labels, keep_prob: 1.0})
			print("Epoch = ", epoch, "Validation_Accuracy = ", validation_accuracy)

	# Save training results
	saver.save(sess, save_file)
	print("Saved trained model")
