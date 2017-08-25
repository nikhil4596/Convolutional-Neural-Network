import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('C:\\tmp\Mnist_data', one_hot=True)

num_classes = 10
num_epochs = 10
batch_size = 128

x = tf.placeholder('float', shape=[None, 784])
y = tf.placeholder('float')


def model(x):
    # Created the Model with 2 Convolutional Layers and one Fully Connected layer

    weights = {'conv_layer_1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'conv_layer_2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'fully_connected': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
               'output': tf.Variable(tf.random_normal([1024, num_classes]))}
    biases = {'conv_layer_1': tf.Variable(tf.random_normal([32])),
              'conv_layer_2': tf.Variable(tf.random_normal([64])),
              'fully_connected': tf.Variable(tf.random_normal([1024])),
              'output': tf.Variable(tf.random_normal([num_classes]))}

    x = tf.reshape(x, shape=[-1,28,28,1])

    conv1 = conv2d(x, weights['conv_layer_1'])
    conv1 = maxpool2d(conv1)
    conv2 = conv2d(conv1, weights['conv_layer_2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['fully_connected']) + biases['fully_connected'])

    output = tf.matmul(fc, weights['output']) + biases['output']
    return output

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def train(x):
    prediction = model(x)
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
