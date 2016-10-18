import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import random

image_size = 28
num_labels = 10
num_channels = 1

with open('test_clf.pickle', 'rb') as f:
  test_dataset, test_labels = pickle.load(f)
  valid_dataset = test_dataset[:10000]
  valid_dataset = valid_dataset.reshape((-1, image_size* image_size)).astype(np.float32)
  valid_labels = test_labels[:10000]
  valid_labels = (np.arange(1, num_labels + 1) == valid_labels[:,None]).astype(np.float32)
  test_dataset = test_dataset[10000:]
  test_dataset = test_dataset.reshape((-1, image_size* image_size)).astype(np.float32)
  test_labels = test_labels[10000:]
  test_labels = (np.arange(1, num_labels + 1) == test_labels[:,None]).astype(np.float32)
    
with open('train_clf.pickle', 'rb') as f:
  train_dataset, train_labels = pickle.load(f)
  train_dataset = train_dataset.reshape((-1, image_size* image_size)).astype(np.float32)
  train_labels = (np.arange(1, num_labels + 1) == train_labels[:,None]).astype(np.float32)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


batch_size = 128
h1 = 1200
h2 = 300
h3 = 75

graph = tf.Graph()
with graph.as_default():

    # Input data. 
    tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    
    # Variables.
    weights_layer1 = tf.Variable(tf.truncated_normal([image_size * image_size, h1], stddev=0.01))
    biases_layer1 = tf.Variable(tf.zeros([h1]))
    weights_layer2 = tf.Variable(tf.truncated_normal([h1, h2], stddev=0.01))
    biases_layer2 = tf.Variable(tf.zeros([h2]))
    weights_layer3 = tf.Variable(tf.truncated_normal([h2, h3], stddev=0.01))
    biases_layer3 = tf.Variable(tf.zeros([h3]))

    # Variables.
    weights = tf.Variable(tf.truncated_normal([h3, num_labels], stddev=0.01))
    biases = tf.Variable(tf.zeros([num_labels]))

    # Load and set check points
    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()
  
    # Model.
    def model(data):
      hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(data, weights_layer1) + biases_layer1), 0.5)
      hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(hidden, weights_layer2) + biases_layer2), 0.5)
      hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(hidden, weights_layer3) + biases_layer3), 0.5)

      return tf.matmul(hidden, weights) + biases
      
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  
    # Optimizer.
    global_step = tf.Variable(0)  # count the number of steps taken.
    learning_rate = tf.train.exponential_decay(0.3, global_step, 5000, 0.9, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
  
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)

    # Evaluation.
    def evaluation(data):
      hidden = tf.nn.relu(tf.matmul(data, weights_layer1) + biases_layer1)
      hidden = tf.nn.relu(tf.matmul(hidden, weights_layer2) + biases_layer2)
      hidden = tf.nn.relu(tf.matmul(hidden, weights_layer3) + biases_layer3)

      return tf.matmul(hidden, weights) + biases
    
    valid_prediction = tf.nn.softmax(evaluation(tf_valid_dataset))
    test_prediction = tf.nn.softmax(evaluation(tf_test_dataset))


num_steps = 200001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  try:
    saver.restore(session, "/tmp/model.ckpt")
    print("Model restored.")
  except:
    print('Initialized')
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = random.randint(0, train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 1000 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
      save_path = saver.save(session, "/tmp/model.ckpt")
      print("Model saved in file: %s" % save_path)
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
