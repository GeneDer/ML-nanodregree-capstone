import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import random

image_size = 64
num_labels = 8
num_channels = 1

#import matplotlib.pyplot as plt

with open('test.pickle', 'rb') as f:
  test_dataset = pickle.load(f).reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
 # plt.imshow(test_dataset[2].reshape(64,64))
  #plt.show()

    
with open('test_labels.pickle', 'rb') as f:
  test_labels = pickle.load(f)
  test_labels['length'] = (np.arange(num_labels) == test_labels['length'][:]).astype(np.float32)
  #print test_labels['length'][2]

with open('train.pickle', 'rb') as f:
  valid_dataset = pickle.load(f).reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
with open('train_labels.pickle', 'rb') as f:
  valid_labels = pickle.load(f)
  valid_labels['length'] = (np.arange(num_labels) == valid_labels['length'][:]).astype(np.float32)
with open('extra.pickle', 'rb') as f:
  train_dataset = pickle.load(f).reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
with open('extra_labels.pickle', 'rb') as f:
  train_labels = pickle.load(f)
  train_labels['length'] = (np.arange(num_labels) == train_labels['length'][:]).astype(np.float32)

print('Training set', train_dataset.shape, train_labels['length'].shape)
print('Validation set', valid_dataset.shape, valid_labels['length'].shape)
print('Test set', test_dataset.shape, test_labels['length'].shape)



def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

batch_size = 16
patch_size = 5
depth = 16
num_hidden1 = 1024
num_hidden2 = 512

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  conv1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth], stddev=0.1),
                               name="conv1_weights")
  conv1_biases = tf.Variable(tf.zeros([depth]),
                               name="conv_biases")
  conv2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1),
                               name="conv2_weights")
  conv2_biases = tf.Variable(tf.constant(1.0, shape=[depth]),
                               name="conv2_biases")
  conv3_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1),
                               name="conv3_weights")
  conv3_biases = tf.Variable(tf.constant(1.0, shape=[depth]),
                               name="conv3_biases")
  conv4_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1),
                               name="conv4_weights")
  conv4_biases = tf.Variable(tf.constant(1.0, shape=[depth]),
                               name="conv4_biases")
  
  hidden1_weights = tf.Variable(tf.truncated_normal(
      [image_size // 16 * image_size // 16 * depth, num_hidden1], stddev=0.1),
                               name="hidden1_weights")
  hidden1_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden1]),
                               name="hidden1_biases")
  hidden2_weights = tf.Variable(tf.truncated_normal(
      [num_hidden1, num_hidden2], stddev=0.1),
                               name="hidden2_weights")
  hidden2_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden2]),
                               name="hidden2_biases")
  
  out_weights = tf.Variable(tf.truncated_normal(
      [num_hidden2, num_labels], stddev=0.1),
                               name="out_weights")
  out_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]),
                               name="out_biases")
  
  init_op = tf.initialize_all_variables()
  saver = tf.train.Saver()

  # Model.
  def model(data):
    conv = tf.nn.relu(tf.nn.conv2d(data,
                                   conv1_weights,
                                   [1, 2, 2, 1],
                                   padding='SAME') + conv1_biases)
    max_pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1],
                              strides=[1, 1, 1, 1], padding='SAME')
    
    conv = tf.nn.relu(tf.nn.conv2d(max_pool,
                                   conv2_weights,
                                   [1, 2, 2, 1],
                                   padding='SAME') + conv2_biases)
    max_pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1],
                              strides=[1, 1, 1, 1], padding='SAME')

    conv = tf.nn.relu(tf.nn.conv2d(max_pool,
                                   conv3_weights,
                                   [1, 2, 2, 1],
                                   padding='SAME') + conv3_biases)
    max_pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1],
                              strides=[1, 1, 1, 1], padding='SAME')

    conv = tf.nn.relu(tf.nn.conv2d(max_pool,
                                   conv4_weights,
                                   [1, 2, 2, 1],
                                   padding='SAME') + conv4_biases)
    max_pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1],
                              strides=[1, 1, 1, 1], padding='SAME')
    
    shape = max_pool.get_shape().as_list()
    reshape = tf.reshape(max_pool, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, hidden1_weights) + hidden1_biases)
    hidden = tf.nn.relu(tf.matmul(hidden, hidden2_weights) + hidden2_biases)
    #for i in xrange(number_length):
        
    
    return tf.matmul(hidden, out_weights) + out_biases
  
  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))


num_steps = 100001

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    try:
        saver.restore(session, "/tmp/model.ckpt")
        print("Model restored.")
    except:
        print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels['length'].shape[0] - batch_size)
        #offset = random.randint(0, train_labels['length'].shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels['length'][offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 100 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels['length']))
            save_path = saver.save(session, "/tmp/model.ckpt")
            print("Model saved in file: %s" % save_path)
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels['length']))
