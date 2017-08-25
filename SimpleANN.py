import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('C:\\tmp\Mnist_data', one_hot=True)

hl1_nodes = 500
hl2_nodes = 500
hl3_nodes = 500
num_classes = 10
num_epochs = 10
batch_size = 100

x = tf.placeholder('float', shape=[None, 784])
y = tf.placeholder('float')


def model(data):
    # Created the Model with 3 hidden layers and one output layer

    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([784, hl1_nodes])),
                      'biases': tf.Variable(tf.random_normal([hl1_nodes]))}
    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([hl1_nodes, hl2_nodes])),
                      'biases': tf.Variable(tf.random_normal([hl2_nodes]))}
    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([hl2_nodes, hl3_nodes])),
                      'biases': tf.Variable(tf.random_normal([hl3_nodes]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([hl3_nodes, num_classes])),
                    'biases': tf.Variable(tf.random_normal([num_classes]))}

    # Feed Forward the Model

    l1 = tf.nn.relu(tf.matmul(data, hidden_layer_1['weights']) + hidden_layer_1['biases'])
    l2 = tf.nn.relu(tf.matmul(l1, hidden_layer_2['weights']) + hidden_layer_2['biases'])
    l3 = tf.nn.relu(tf.matmul(l2, hidden_layer_3['weights']) + hidden_layer_3['biases'])
    output = (tf.matmul(l3, output_layer['weights']) + output_layer['biases'])

    return output


def train(data):
    prediction = model(data)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                x1, y1 = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: x1, y: y1})
                epoch_loss += c
            print("Epoch ", epoch, ' completed out of ', num_epochs, 'with epoch loss: ', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train(x)
