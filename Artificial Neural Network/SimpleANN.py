import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('C:\tmp\Mnist_data', one_hot=True)

hl1_nodes = 500
hl2_nodes = 500
hl3_nodes = 500
num_classes = 10
num_epochs = 10
batch_size = 100

x = tf.placeholder('float',shape=[None, 784])
y = tf.placeholder('float')

def model(data):
    hidden_layer_1 = {'weigths': tf.variable(tf.random_normal([784, hl1_nodes])),
                      'biases': tf.variable(tf.random_normal([hl1_nodes]))}