import tensorflow as tf
import json
from scipy.misc import imread, imresize
import numpy as np

from train import build_forward
from utils import googlenet_load, train_utils
from utils.train_utils import add_rectangles

import time
init_time = time.time()

hypes_file = 'overfeat_rezoom.json'
with open(hypes_file, 'r') as f:
  H = json.load(f)

### Need to change the ckpt path ###
ckpt_path = "overfeat_checkpint.ckpt"

### Need to change image path ###
image_path = "project.png"

image_data = imread(image_path, mode = "RGB")
image_height, image_width, image_depth = image_data.shape
image_data = imresize(image_data, (480, 640))

image_data_gray = imread(image_path, flatten = True)
image_data_gray = imresize(image_data_gray, (480, 640))

from PIL import Image, ImageDraw, ImageFont
im = Image.open(image_path)
im = im.resize((640, 480))
#im.show()

image_size = 28
num_labels = 10
def number_classification(region_of_interest):
    h1 = 1200
    h2 = 300
    h3 = 75

    graph = tf.Graph()
    with graph.as_default():

        # Input data. 
        tf_test_data = tf.constant(region_of_interest)
    
        # Variables.
        weights_layer1 = tf.Variable(tf.truncated_normal([image_size * image_size, h1], stddev=0.01))
        biases_layer1 = tf.Variable(tf.zeros([h1]))
        weights_layer2 = tf.Variable(tf.truncated_normal([h1, h2], stddev=0.01))
        biases_layer2 = tf.Variable(tf.zeros([h2]))
        weights_layer3 = tf.Variable(tf.truncated_normal([h2, h3], stddev=0.01))
        biases_layer3 = tf.Variable(tf.zeros([h3]))
        weights = tf.Variable(tf.truncated_normal([h3, num_labels], stddev=0.01))
        biases = tf.Variable(tf.zeros([num_labels]))

        # Load and set check points
        saver = tf.train.Saver()

        # Evaluation.
        def evaluation(data):
            hidden = tf.nn.relu(tf.matmul(data, weights_layer1) + biases_layer1)
            hidden = tf.nn.relu(tf.matmul(hidden, weights_layer2) + biases_layer2)
            hidden = tf.nn.relu(tf.matmul(hidden, weights_layer3) + biases_layer3)

            return tf.matmul(hidden, weights) + biases
    
        test_prediction = tf.nn.softmax(evaluation(tf_test_data))

    with tf.Session(graph=graph) as session:
        
        saver.restore(session, "classification_model.ckpt")
        pred = test_prediction.eval()
        num = (1+np.argmax(pred))%10
        #print "The number is:", num
        return num

tf.reset_default_graph()
googlenet = googlenet_load.init(H)
x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
if H['use_rezoom']:
    pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_forward(H, tf.expand_dims(x_in, 0), googlenet, 'test', reuse=None)
    grid_area = H['grid_height'] * H['grid_width']
    pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], 2])), [grid_area, H['rnn_len'], 2])
    if H['reregress']:
        pred_boxes = pred_boxes + pred_boxes_deltas
else:
    pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), googlenet, 'test', reuse=None)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, ckpt_path)
    
    feed = {x_in: image_data}
    (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)
    new_img, rects = add_rectangles(H, [image_data], np_pred_confidences, np_pred_boxes,
                                    use_stitching=True, rnn_len=H['rnn_len'], min_conf=0.3)
    draw = ImageDraw.Draw(im)
    for rec in rects:
      if rec.score > 0.2:
        #print rec.x1, rec.y1, rec.x2, rec.y2, rec.score
        number = imresize(image_data_gray[int(rec.y1):int(rec.y2),
                                          int(rec.x1):int(rec.x2)], (image_size, image_size))
        region_of_interest = (number.astype(float) - 255 / 2) / 255
        num = number_classification(region_of_interest.reshape((-1, image_size* image_size)).astype(np.float32))

        draw.rectangle([rec.x1, rec.y1, rec.x2, rec.y2])
        draw.text([rec.x1+2, rec.y1+2], str(num))
        
    del draw
    #im = im.resize((image_width, image_height))
    im.save("xyz.PNG")

print "Total time used: ", time.time() - init_time
