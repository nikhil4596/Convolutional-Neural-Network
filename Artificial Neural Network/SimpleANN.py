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

    # Created the Model with 3 hidden layers and one output layer

    hidden_layer_1 = {'weigths': tf.variable(tf.random_normal([784, hl1_nodes])),
                      'biases': tf.variable(tf.random_normal([hl1_nodes]))}
    hidden_layer_2 = {'weigths': tf.variable(tf.random_normal([hl1_nodes, hl2_nodes])),
                      'biases': tf.variable(tf.random_normal([hl2_nodes]))}
    hidden_layer_3 = {'weigths': tf.variable(tf.random_normal([hl2_nodes, hl3_nodes])),
                      'biases': tf.variable(tf.random_normal([hl3_nodes]))}
    output_layer = {'weigths': tf.variable(tf.random_normal([hl3_nodes, num_classes])),

                      'biases': tf.variable(tf.random_normal([num_classes]))}

    # Feed Forward the Model

    l1 = tf.nn.relu(tf.matmul(data, hidden_layer_1['weigths']) + hidden_layer_1['biases'])
    l2 = tf.nn.relu(tf.matmul(l1, hidden_layer_2['weigths']) + hidden_layer_2['biases'])
    l3 = tf.nn.relu(tf.matmul(l2, hidden_layer_3['weigths']) + hidden_layer_3['biases'])
    output = tf.nn.relu(tf.matmul(l3, hidden_layer_3['weigths']) + hidden_layer_3['biases'])

    return output

def train(data):
    pass
